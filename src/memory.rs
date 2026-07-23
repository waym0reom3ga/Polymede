use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::{SqlitePool, Row};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

use crate::config::{LlmConfig, MemoryConfig};
use crate::llm::{LlmClient, Message, MessageRole};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Raw agent interaction captured at Layer 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawInteraction {
    pub id: Uuid,
    pub timestamp: chrono::DateTime<Utc>,
    pub session_id: String,
    pub chunk_id: String,
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<String>,
    pub tags: Vec<String>,
}

/// Compressed memory living at Layer 1+.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMemory {
    pub id: Uuid,
    pub level: u32,
    pub summary: String,
    pub tags: Vec<String>,
    pub source_count: u32,
    pub token_estimate: u32,
    pub created_at: chrono::DateTime<Utc>,
}

/// Result of a recall query, already filtered within token budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub memories: Vec<CompressedMemory>,
    pub raw_interactions: Vec<RawInteraction>,
    pub tokens_used: u32,
    pub tokens_budget: u32,
}

/// Snapshot of memory subsystem state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatus {
    pub raw_count: u64,
    pub compressed_count: u64,
    pub highest_level: u32,
    pub last_compression: Option<chrono::DateTime<Utc>>,
    pub pending_raw: u64,
    pub next_chunk_id: u64,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum MemoryError {
    Db(String),
    Config(String),
    Compression(String),
    Io(String),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::Db(msg) => write!(f, "memory db error: {msg}"),
            MemoryError::Config(msg) => write!(f, "memory config error: {msg}"),
            MemoryError::Compression(msg) => write!(f, "compression error: {msg}"),
            MemoryError::Io(msg) => write!(f, "memory IO error: {msg}"),
        }
    }
}

impl std::error::Error for MemoryError {}

// ---------------------------------------------------------------------------
// TotalRecall integration layer
// ---------------------------------------------------------------------------

pub struct MemoryIntegration {
    /// Layer 0 store for raw agent interactions.
    log_pool: SqlitePool,
    /// Layer 1+ store for compressed memories.
    recall_pool: SqlitePool,
    /// Shared config for token budgeting and scheduling.
    config: MemoryConfig,
    /// LLM config for distillation calls (None = mechanical fallback only).
    llm_config: Option<LlmConfig>,
    /// Current session identifier.
    session_id: String,
    /// Auto-compression timer handle (None when disabled).
    _compress_timer: Option<tokio::task::JoinHandle<()>>,
}

impl MemoryIntegration {
    /// Build the integration layer, connecting to both databases.
    ///
    /// `state_dir` is the parent directory for `memory_log.db` and
    /// `total_recall.db`.  If `auto_compress` is true a background task
    /// will run compression on the configured interval.
    /// Pass an `LlmConfig` to enable LLM-assisted distillation; pass
    /// `None` to fall back to mechanical summaries only.
    pub async fn new(
        state_dir: PathBuf,
        config: MemoryConfig,
        llm_config: Option<LlmConfig>,
        auto_compress: bool,
    ) -> Result<Self, MemoryError> {
        let log_path = state_dir.join("memory_log.db");
        let recall_path = state_dir.join("total_recall.db");

        std::fs::create_dir_all(&state_dir).map_err(|e| {
            MemoryError::Io(format!("cannot create state dir {:?}: {e}", state_dir))
        })?;

        let log_pool = Self::connect(&log_path).await?;
        let recall_pool = Self::connect(&recall_path).await?;

        Self::migrate_log(&log_pool).await?;
        Self::migrate_recall(&recall_pool).await?;

        let session_id = Uuid::new_v4().to_string();

        let _compress_timer = if auto_compress {
            let log_clone = sqlx::pool::Pool::clone(&log_pool);
            let recall_clone = sqlx::pool::Pool::clone(&recall_pool);
            let cfg = config.clone();
            let llm_cfg = llm_config.clone();

            Some(tokio::spawn(async move {
                Self::compression_loop(log_clone, recall_clone, cfg, llm_cfg).await
            }))
        } else {
            None
        };

        tracing::info!(
            "memory integration ready (log={:?}, recall={:?})",
            log_path,
            recall_path
        );

        Ok(Self {
            log_pool,
            recall_pool,
            config,
            llm_config,
            session_id,
            _compress_timer,
        })
    }

    // -- public API ---------------------------------------------------------

    /// Ingest a raw agent interaction into Layer 0.
    ///
    /// The `chunk_id` is auto-assigned.  Tags are stored as-is for
    /// later recall matching.
    pub async fn ingest(
        &self,
        role: &str,
        content: &str,
        tool_calls: &[String],
        tags: &[String],
    ) -> Result<Uuid, MemoryError> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let chunk_id = self.next_chunk_id().await;

        sqlx::query(
            r#"
            INSERT INTO raw_interactions
                (id, timestamp, session_id, chunk_id, role, content, tool_calls, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(id.to_string())
        .bind(now.to_rfc3339())
        .bind(&self.session_id)
        .bind(chunk_id.to_string())
        .bind(role)
        .bind(content)
        .bind(serde_json::to_string(tool_calls).map_err(|e| MemoryError::Db(e.to_string()))?)
        .bind(serde_json::to_string(tags).map_err(|e| MemoryError::Db(e.to_string()))?)
        .execute(&self.log_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        tracing::debug!(
            interaction_id = %id,
            chunk = %chunk_id,
            "ingested raw interaction"
        );

        Ok(id)
    }

    /// Trigger a compression cycle: distill uncompressed L0 interactions
    /// into L1 memories and recursively merge into higher levels.
    pub async fn compress(&self) -> Result<u32, MemoryError> {
        let pending = self.pending_raw_count().await?;
        if pending == 0 {
            tracing::debug!("no pending interactions to compress");
            return Ok(0);
        }

        let chunks = self.uncompressed_chunks().await?;
        let mut produced = 0u32;

        for chunk in &chunks {
            let interactions = self.interactions_for_chunk(chunk).await?;
            if interactions.is_empty() {
                continue;
            }

            let summary = self.distill_chunk(&interactions).await?;
            let tokens = estimate_tokens(&summary);

            self.store_compressed(
                1,
                &summary,
                &interactions.first().unwrap().tags,
                interactions.len() as u32,
                tokens,
            )
            .await?;

            self.mark_chunk_compressed(chunk).await?;
            produced += 1;
        }

        if produced > 0 {
            self.merge_higher_levels().await?;
        }

        tracing::info!(produced = produced, "compression cycle complete");
        Ok(produced)
    }

    /// Recall memories relevant to `query_tags` within `token_budget`.
    ///
    /// Returns both compressed memories (preferred) and recent raw
    /// interactions, sorted by relevance and clamped to budget.
    pub async fn recall(
        &self,
        query_tags: &[String],
        token_budget: Option<u32>,
    ) -> Result<RecallResult, MemoryError> {
        let budget = token_budget.unwrap_or(self.config.max_recall_tokens as u32);

        let compressed = self.relevant_compressed(query_tags).await?;
        let raw = self.relevant_raw(query_tags).await?;

        let mut result = RecallResult {
            memories: Vec::new(),
            raw_interactions: Vec::new(),
            tokens_used: 0,
            tokens_budget: budget,
        };

        for mem in compressed {
            let tok = mem.token_estimate;
            if result.tokens_used + tok > budget {
                break;
            }
            result.memories.push(mem);
            result.tokens_used += tok;
        }

        for interaction in raw {
            let size = estimate_tokens(&interaction.content);
            if result.tokens_used + size > budget {
                break;
            }
            result.raw_interactions.push(interaction);
            result.tokens_used += size;
        }

        tracing::debug!(
            tokens_used = result.tokens_used,
            budget = budget,
            compressed = result.memories.len(),
            raw = result.raw_interactions.len(),
            "recall complete"
        );

        Ok(result)
    }

    /// Return a status snapshot of the memory subsystem.
    pub async fn status(&self) -> Result<MemoryStatus, MemoryError> {
        let raw_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM raw_interactions",
        )
        .fetch_one(&self.log_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        let compressed_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM compressed_memories",
        )
        .fetch_one(&self.recall_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        let highest_level: i64 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(level), 0) FROM compressed_memories",
        )
        .fetch_one(&self.recall_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        let last_compression: Option<String> = sqlx::query_scalar(
            "SELECT created_at FROM compressed_memories ORDER BY created_at DESC LIMIT 1",
        )
        .fetch_one(&self.recall_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))
        .ok()
        .flatten();

        let pending_raw: u64 = self.pending_raw_count().await?;
        let next_chunk_id: u64 = self.next_chunk_id().await;

        Ok(MemoryStatus {
            raw_count: raw_count as u64,
            compressed_count: compressed_count as u64,
            highest_level: highest_level as u32,
            last_compression: last_compression
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc)),
            pending_raw,
            next_chunk_id,
        })
    }

    /// Return the current session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    // -- database helpers ---------------------------------------------------

    async fn connect(path: &std::path::Path) -> Result<SqlitePool, MemoryError> {
        let url = format!("sqlite:{}", path.display());
        SqlitePool::connect(&url)
            .await
            .map_err(|e| MemoryError::Db(format!("cannot connect to {:?}: {e}", path)))
    }

    async fn migrate_log(pool: &SqlitePool) -> Result<(), MemoryError> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS raw_interactions (
                id          TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                chunk_id    TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                tool_calls  TEXT NOT NULL DEFAULT '[]',
                tags        TEXT NOT NULL DEFAULT '[]',
                compressed  INTEGER NOT NULL DEFAULT 0
            )"#,
        )
        .execute(pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        sqlx::query(
            r#"CREATE INDEX IF NOT EXISTS idx_raw_chunk
               ON raw_interactions(chunk_id, compressed)"#,
        )
        .execute(pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        sqlx::query(
            r#"CREATE INDEX IF NOT EXISTS idx_raw_session
               ON raw_interactions(session_id, timestamp)"#,
        )
        .execute(pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        Ok(())
    }

    async fn migrate_recall(pool: &SqlitePool) -> Result<(), MemoryError> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS compressed_memories (
                id             TEXT PRIMARY KEY,
                level          INTEGER NOT NULL,
                summary        TEXT NOT NULL,
                tags           TEXT NOT NULL DEFAULT '[]',
                source_count   INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL
            )"#,
        )
        .execute(pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        sqlx::query(
            r#"CREATE INDEX IF NOT EXISTS idx_compressed_level
               ON compressed_memories(level, created_at)"#,
        )
        .execute(pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        Ok(())
    }

    // -- chunk management ---------------------------------------------------

    async fn pending_raw_count(&self) -> Result<u64, MemoryError> {
        let row: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM raw_interactions WHERE compressed = 0",
        )
        .fetch_one(&self.log_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;
        Ok(row.0 as u64)
    }

    async fn next_chunk_id(&self) -> u64 {
        let row: Option<(i64,)> = sqlx::query_as(
            "SELECT MAX(CAST(chunk_id AS INTEGER)) + 1 FROM raw_interactions",
        )
        .fetch_optional(&self.log_pool)
        .await
        .unwrap_or(None);
        row.map(|r| r.0 as u64).unwrap_or(1)
    }

    async fn uncompressed_chunks(&self) -> Result<Vec<String>, MemoryError> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT chunk_id FROM raw_interactions WHERE compressed = 0",
        )
        .fetch_all(&self.log_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;
        Ok(rows.into_iter().map(|r| r.0).collect())
    }

    async fn interactions_for_chunk(
        &self,
        chunk: &str,
    ) -> Result<Vec<RawInteraction>, MemoryError> {
        let rows = sqlx::query(
            r#"
            SELECT id, timestamp, session_id, chunk_id, role, content, tool_calls, tags
            FROM raw_interactions
            WHERE chunk_id = ?
            ORDER BY timestamp ASC
            "#,
        )
        .bind(chunk)
        .map(|r: sqlx::sqlite::SqliteRow| {
            RawInteraction {
                id: r.try_get("id").ok().and_then(|s: String| Uuid::parse_str(&s).ok()).unwrap_or_default(),
                timestamp: r.try_get("timestamp").ok().and_then(|s: String| chrono::DateTime::parse_from_rfc3339(&s).ok()).map(|dt| dt.with_timezone(&chrono::Utc)).unwrap_or_default(),
                session_id: r.try_get("session_id").unwrap_or_default(),
                chunk_id: r.try_get("chunk_id").unwrap_or_default(),
                role: r.try_get("role").unwrap_or_default(),
                content: r.try_get("content").unwrap_or_default(),
                tool_calls: r.try_get("tool_calls").ok().and_then(|s: String| serde_json::from_str(&s).ok()).unwrap_or_default(),
                tags: r.try_get("tags").ok().and_then(|s: String| serde_json::from_str(&s).ok()).unwrap_or_default(),
            }
        })
        .fetch_all(&self.log_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        Ok(rows)
    }

    async fn mark_chunk_compressed(&self, chunk: &str) -> Result<(), MemoryError> {
        sqlx::query(
            "UPDATE raw_interactions SET compressed = 1 WHERE chunk_id = ?",
        )
        .bind(chunk)
        .execute(&self.log_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;
        Ok(())
    }

    // -- distillation -------------------------------------------------------

    /// Distill a chunk of raw interactions into a compressed memory entry.
    /// Uses the LLM for intelligent summarization when available, falls back
    /// to mechanical keyword extraction on error or when no LLM config is set.
    async fn distill_chunk(
        &self,
        interactions: &[RawInteraction],
    ) -> Result<String, MemoryError> {
        let text = interactions.iter()
            .map(|i| format!("[{}]: {}", i.role, i.content))
            .collect::<Vec<_>>()
            .join("\n");

        // Try LLM-assisted distillation first
        if let Some(ref _llm_cfg) = self.llm_config {
            match self.distill_with_llm(&text).await {
                Ok(summary) => {
                    tracing::info!("LLM distillation succeeded");
                    return Ok(summary);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "LLM distillation failed, falling back to mechanical summary");
                }
            }
        }

        // Mechanical fallback: keyword extraction + frequency analysis
        let (summary, _tags) = Self::mechanical_summary(&text);
        Ok(summary)
    }

    /// Call the LLM to distill raw interactions into a compressed memory.
    async fn distill_with_llm(
        &self,
        text: &str,
    ) -> Result<String, MemoryError> {
        let client = LlmClient::new(self.llm_config.as_ref().unwrap().clone());

        // Truncate input if too long (keep last 80% to preserve recent context)
        let max_input_len = 60_000;
        let truncated = if text.len() > max_input_len {
            &text[text.len().saturating_sub(max_input_len)..]
        } else {
            text
        };

        let prompt = format!(
            r#"You are a memory compression system. Your job is to distill the following conversation log into concise, actionable memories.

Conversation log:
---
{truncated}
---

Respond with ONLY a bulleted list of key facts, decisions, and important context extracted from this conversation. One fact per bullet point. No preamble, no markdown formatting other than bullets. If nothing worth remembering is found, write "No significant memories to extract."

Rules:
- Extract only factual information, decisions made, and important context.
- Each memory should be a standalone fact that would help answer future questions.
- Omit pleasantries, greetings, and filler conversation.
- Be specific — include names, values, file paths, error messages when present."#
        );

        let messages = vec![Message {
            role: MessageRole::User,
            content: prompt,
            tool_calls: None,
            tool_call_id: None,
        }];

        let response = client.chat(messages, None).await.map_err(|e| MemoryError::Io(format!("LLM distillation call failed: {e}")))?;

        Ok(response.content)
    }

    /// Mechanical fallback: extract keywords and build a frequency-based summary.
    fn mechanical_summary(text: &str) -> (String, Vec<String>) {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for word in &words {
            if word.len() > 3 {
                *freq.entry(word.to_lowercase()).or_insert(0) += 1;
            }
        }

        let mut top: Vec<(&String, &usize)> = freq.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1));
        let keywords: Vec<String> = top.iter().take(10).map(|(w, _)| (*w).clone()).collect();

        // Build summary from most frequent sentences
        let sentences: Vec<&str> = text.split('.').filter(|s| !s.trim().is_empty()).collect();
        let summary = if sentences.is_empty() {
            format!("Distilled {} words, keywords: {}", words.len(), keywords.join(", "))
        } else {
            // Pick the 3 most content-rich sentences (by keyword overlap)
            let mut scored: Vec<(usize, &&str)> = sentences.iter().enumerate().map(|(_i, s)| {
                let score = keywords.iter().filter(|k| s.to_lowercase().contains(k.as_str())).count();
                (score, s)
            }).collect();
            scored.sort_by(|a, b| b.0.cmp(&a.0));
            let top_sentences: Vec<String> = scored.iter().take(3).map(|(_, s)| (*s).trim().to_string()).collect();
            format!("{} | Keywords: {}", top_sentences.join(". "), keywords.join(", "))
        };

        (summary, keywords)
    }

    async fn store_compressed(
        &self,
        level: u32,
        summary: &str,
        tags: &[String],
        source_count: u32,
        token_estimate: u32,
    ) -> Result<(), MemoryError> {
        let id = Uuid::new_v4();
        let now = Utc::now();

        sqlx::query(
            r#"
            INSERT INTO compressed_memories
                (id, level, summary, tags, source_count, token_estimate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(id.to_string())
        .bind(level as i64)
        .bind(summary)
        .bind(serde_json::to_string(tags).map_err(|e| MemoryError::Db(e.to_string()))?)
        .bind(source_count as i64)
        .bind(token_estimate as i64)
        .bind(now.to_rfc3339())
        .execute(&self.recall_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        Ok(())
    }

    /// Recursively merge memories at a given level into higher-level
    /// abstractions when enough have accumulated. Uses LLM-assisted merging
    /// when available, falls back to concatenation on error.
    async fn merge_higher_levels(&self) -> Result<(), MemoryError> {
        // Fetch memories at level >= 5 for merging
        let candidates = sqlx::query_as::<_, (String, String, i32)>(
            "SELECT content, tags, level FROM compressed_memories WHERE level >= 5 ORDER BY relevance_score DESC",
        )
        .fetch_all(&self.recall_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        if candidates.len() < 2 {
            return Ok(());
        }
        // Try LLM-assisted merge first
        let memories_text = candidates.iter().map(|(c, _, _)| c.as_str()).collect::<Vec<_>>().join("\n---\n");

        if let Some(ref _llm_cfg) = self.llm_config {
            match self.merge_with_llm(&memories_text).await {
                Ok(merged_memories) => {
                    let merged_count = merged_memories.len();
                    // Delete old candidates and store merged result
                    for (content, tags, level) in &candidates {
                        let _ = sqlx::query("DELETE FROM compressed_memories WHERE content = ? AND tags = ? AND level = ?")
                            .bind(content.as_str())
                            .bind(tags.as_str())
                            .bind(*level as i64)
                            .execute(&self.recall_pool)
                            .await;
                    }

                    for (summary, tags) in merged_memories {
                        let tokens = estimate_tokens(&summary);
                        let _ = self.store_compressed(1, &summary, &tags, 0, tokens).await;
                    }

                    tracing::info!(merged = merged_count, "LLM merge succeeded");
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!(error = %e, "LLM merge failed, falling back to concatenation");
                }
            }
        }

        // Fallback: simple concatenation with separator
        let merged = candidates.iter().map(|(c, _, _)| c.as_str()).collect::<Vec<_>>().join(" | ");
        if !merged.is_empty() {
            let tokens = estimate_tokens(&merged);
            self.store_compressed(1, &merged, &["merged".to_string()], 0, tokens).await?;
        }

        Ok(())
    }

    /// Call the LLM to merge multiple high-level memories into fewer, more general ones.
    async fn merge_with_llm(
        &self,
        memories_text: &str,
    ) -> Result<Vec<(String, Vec<String>)>, MemoryError> {
        let client = LlmClient::new(self.llm_config.as_ref().unwrap().clone());

        // Truncate if too long
        let max_input_len = 40_000;
        let truncated = if memories_text.len() > max_input_len {
            &memories_text[memories_text.len().saturating_sub(max_input_len)..]
        } else {
            memories_text
        };

        let prompt = format!(
            r#"You are merging multiple existing compressed memories into fewer, more general ones. The goal is to reduce redundancy while preserving all unique information.

Current memories:
---
{truncated}
---

Respond in exactly this JSON format (no markdown, no extra text):
[
  {{
    "summary": "A concise merged memory capturing the key facts",
    "tags": ["tag1", "tag2"]
  }}
]

Rules:
- Merge related memories into single entries.
- Preserve all unique information — do not drop facts.
- Each entry should be a standalone, self-contained fact.
- Tags should be lowercase keywords relevant to the content."#
        );

        let messages = vec![Message {
            role: MessageRole::User,
            content: prompt,
            tool_calls: None,
            tool_call_id: None,
        }];

        let response = client.chat(messages, None).await.map_err(|e| MemoryError::Io(format!("LLM merge call failed: {e}")))?;

        // Parse JSON array
        let parsed: serde_json::Value = serde_json::from_str(&response.content)
            .map_err(|e| MemoryError::Io(format!("Failed to parse LLM merge output as JSON: {e}")))?;

        let memories_array = parsed.as_array()
            .ok_or_else(|| MemoryError::Io("LLM merge response is not a JSON array".to_string()))?;

        let mut result = Vec::new();
        for mem in memories_array {
            if let (Some(summary), Some(tags)) = (
                mem.get("summary").and_then(|s| s.as_str()),
                mem.get("tags").and_then(|t| t.as_array()),
            ) {
                let tag_vec: Vec<String> = tags.iter().filter_map(|t| t.as_str()).map(|t| t.to_lowercase()).collect();
                result.push((summary.to_string(), tag_vec));
            }
        }

        if result.is_empty() {
            return Ok(vec![(memories_text.lines().take(5).collect::<Vec<_>>().join("\n"), vec!["merged".to_string()])]);
        }

        Ok(result)
    }

    // -- recall helpers -----------------------------------------------------

    async fn relevant_compressed(
        &self,
        query_tags: &[String],
    ) -> Result<Vec<CompressedMemory>, MemoryError> {
        let rows = Self::fetch_all_compressed(&self.recall_pool).await?;

        if query_tags.is_empty() {
            return Ok(rows);
        }

        let matched: Vec<CompressedMemory> = rows
            .into_iter()
            .filter(|m| Self::tags_overlap(&m.tags, query_tags))
            .collect();

        Ok(matched)
    }

    async fn relevant_raw(
        &self,
        query_tags: &[String],
    ) -> Result<Vec<RawInteraction>, MemoryError> {
        let rows = Self::fetch_compressed_raw(&self.log_pool).await?;

        if query_tags.is_empty() {
            return Ok(rows);
        }

        let matched: Vec<RawInteraction> = rows
            .into_iter()
            .filter(|i| Self::tags_overlap(&i.tags, query_tags))
            .collect();

        Ok(matched)
    }

    // -- static row fetchers ------------------------------------------------

    async fn fetch_all_compressed(
        pool: &SqlitePool,
    ) -> Result<Vec<CompressedMemory>, MemoryError> {
        let rows = sqlx::query(
            r#"
            SELECT id, level, summary, tags, source_count, token_estimate, created_at
            FROM compressed_memories
            ORDER BY level DESC, created_at DESC
            "#,
        )
        .map(|r: sqlx::sqlite::SqliteRow| {
            CompressedMemory {
                id: r.try_get("id").ok().and_then(|s: String| Uuid::parse_str(&s).ok()).unwrap_or_default(),
                level: r.try_get("level").unwrap_or(0) as u32,
                summary: r.try_get("summary").unwrap_or_default(),
                tags: r.try_get("tags").ok().and_then(|s: String| serde_json::from_str(&s).ok()).unwrap_or_default(),
                source_count: r.try_get("source_count").unwrap_or(0) as u32,
                token_estimate: r.try_get("token_estimate").unwrap_or(0) as u32,
                created_at: r.try_get("created_at").ok().and_then(|s: String| chrono::DateTime::parse_from_rfc3339(&s).ok()).map(|dt| dt.with_timezone(&chrono::Utc)).unwrap_or_default(),
            }
        })
        .fetch_all(pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        Ok(rows)
    }

    async fn fetch_compressed_at_level(
        pool: &SqlitePool,
        level: u32,
    ) -> Result<Vec<CompressedMemory>, MemoryError> {
        let rows = sqlx::query(
            r#"
            SELECT id, level, summary, tags, source_count, token_estimate, created_at
            FROM compressed_memories
            WHERE level = ?
            ORDER BY created_at DESC
            "#,
        )
        .bind(level as i64)
        .map(|r: sqlx::sqlite::SqliteRow| {
            CompressedMemory {
                id: r.try_get("id").ok().and_then(|s: String| Uuid::parse_str(&s).ok()).unwrap_or_default(),
                level: r.try_get("level").unwrap_or(0) as u32,
                summary: r.try_get("summary").unwrap_or_default(),
                tags: r.try_get("tags").ok().and_then(|s: String| serde_json::from_str(&s).ok()).unwrap_or_default(),
                source_count: r.try_get("source_count").unwrap_or(0) as u32,
                token_estimate: r.try_get("token_estimate").unwrap_or(0) as u32,
                created_at: r.try_get("created_at").ok().and_then(|s: String| chrono::DateTime::parse_from_rfc3339(&s).ok()).map(|dt| dt.with_timezone(&chrono::Utc)).unwrap_or_default(),
            }
        })
        .fetch_all(pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        Ok(rows)
    }

    async fn fetch_compressed_raw(pool: &SqlitePool) -> Result<Vec<RawInteraction>, MemoryError> {
        let rows = sqlx::query(
            r#"
            SELECT id, timestamp, session_id, chunk_id, role, content, tool_calls, tags
            FROM raw_interactions
            WHERE compressed = 1
            ORDER BY timestamp DESC
            LIMIT 100
            "#,
        )
        .map(|r: sqlx::sqlite::SqliteRow| {
            RawInteraction {
                id: r.try_get("id").ok().and_then(|s: String| Uuid::parse_str(&s).ok()).unwrap_or_default(),
                timestamp: r.try_get("timestamp").ok().and_then(|s: String| chrono::DateTime::parse_from_rfc3339(&s).ok()).map(|dt| dt.with_timezone(&chrono::Utc)).unwrap_or_default(),
                session_id: r.try_get("session_id").unwrap_or_default(),
                chunk_id: r.try_get("chunk_id").unwrap_or_default(),
                role: r.try_get("role").unwrap_or_default(),
                content: r.try_get("content").unwrap_or_default(),
                tool_calls: r.try_get("tool_calls").ok().and_then(|s: String| serde_json::from_str(&s).ok()).unwrap_or_default(),
                tags: r.try_get("tags").ok().and_then(|s: String| serde_json::from_str(&s).ok()).unwrap_or_default(),
            }
        })
        .fetch_all(pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        Ok(rows)
    }

    // -- background compression loop ----------------------------------------

    async fn compression_loop(
        log_pool: SqlitePool,
        recall_pool: SqlitePool,
        config: MemoryConfig,
        llm_config: Option<LlmConfig>,
    ) {
        let interval_dur =
            Self::parse_interval(&config.compression_interval).unwrap_or(Duration::from_secs(3600));
        let mut interval = tokio::time::interval(interval_dur);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;

            let compressor = TempCompressor::new(log_pool.clone(), recall_pool.clone(), config.clone(), llm_config.clone());

            match compressor.run().await {
                Ok(count) => {
                    if count > 0 {
                        tracing::info!(produced = count, "background compression cycle");
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "background compression failed");
                }
            }
        }
    }

    // -- tag and token utilities --------------------------------------------

    fn dedup_tags<'a>(interactions: &'a [RawInteraction]) -> String {
        let mut seen = HashSet::new();
        interactions
            .iter()
            .flat_map(|i| i.tags.iter())
            .filter(|t| seen.insert(t.clone()))
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn dedup_from_memories(memories: &[CompressedMemory]) -> Vec<String> {
        let mut seen = HashSet::new();
        memories
            .iter()
            .flat_map(|m| m.tags.iter())
            .filter(|t| seen.insert(t.clone()))
            .cloned()
            .collect()
    }

    fn tags_overlap<'a>(
        stored: &'a [String],
        query: &'a [String],
    ) -> bool {
        if stored.is_empty() || query.is_empty() {
            return false;
        }
        let query_set: HashSet<&str> = query.iter().map(|s| s.as_str()).collect();
        stored.iter().any(|t| query_set.contains(t.as_str()))
    }

    fn parse_interval(s: &str) -> Option<Duration> {
        let s = s.trim();
        let (num, unit) = match s.chars().rev().position(|c| c.is_ascii_digit()) {
            Some(pos) => {
                let unit = &s[..s.len() - 1 - pos];
                let num_str = &s[s.len() - pos..];
                (num_str.parse::<u64>().ok()?, unit)
            }
            None => return None,
        };

        let secs = match unit {
            "s" => num,
            "m" => num * 60,
            "h" => num * 3600,
            "d" => num * 86400,
            _ => return None,
        };

        Some(Duration::from_secs(secs))
    }
}

// ---------------------------------------------------------------------------
// Lightweight compressor for the background task
// ---------------------------------------------------------------------------

struct TempCompressor {
    log_pool: SqlitePool,
    recall_pool: SqlitePool,
    config: MemoryConfig,
    llm_config: Option<LlmConfig>,
}

impl TempCompressor {
    pub fn new(
        log_pool: SqlitePool,
        recall_pool: SqlitePool,
        config: MemoryConfig,
        llm_config: Option<LlmConfig>,
    ) -> Self {
        Self {
            log_pool,
            recall_pool,
            config,
            llm_config,
        }
    }

    async fn run(&self) -> Result<u32, MemoryError> {
        let interactions = MemoryIntegration::fetch_compressed_raw(&self.log_pool).await?;
        if interactions.is_empty() {
            return Ok(0);
        }

        // Try LLM-assisted distillation first
        let summary = if let Some(ref llm_cfg) = self.llm_config {
            match Self::distill_with_llm(llm_cfg, &interactions).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "TempCompressor LLM distillation failed, falling back");
                    Self::mechanical_summary_text(&interactions)
                }
            }
        } else {
            Self::mechanical_summary_text(&interactions)
        };

        let tags = Self::extract_tags(&summary);
        self.store(summary, tags).await?;

        Ok(interactions.len() as u32)
    }

    /// LLM-assisted distillation for TempCompressor.
    async fn distill_with_llm(
        llm_cfg: &LlmConfig,
        interactions: &[RawInteraction],
    ) -> Result<String, MemoryError> {
        let client = LlmClient::new(llm_cfg.clone());

        let text = format!(
            "{}",
            interactions.iter()
                .map(|i| format!("[{}]: {}", i.role, i.content))
                .collect::<Vec<_>>()
                .join("\n")
        );

        // Truncate if too long
        let max_input_len = 60_000;
        let truncated = if text.len() > max_input_len {
            &text[text.len().saturating_sub(max_input_len)..]
        } else {
            &text
        };

        let prompt = format!(
            r#"You are a memory compression system. Distill the following conversation log into concise, actionable memories.

Conversation log:
---
{truncated}
---

Respond with ONLY a bulleted list of key facts, decisions, and important context. One fact per bullet point. No preamble, no markdown formatting other than bullets."#
        );

        let messages = vec![Message {
            role: MessageRole::User,
            content: prompt,
            tool_calls: None,
            tool_call_id: None,
        }];

        let response = client.chat(messages, None).await.map_err(|e| MemoryError::Io(format!("TempCompressor LLM call failed: {e}")))?;

        Ok(response.content)
    }

    /// Mechanical fallback for TempCompressor.
    fn mechanical_summary_text(interactions: &[RawInteraction]) -> String {
        format!(
            "Distilled {} interactions: {} tool calls, topics: {}",
            interactions.len(),
            interactions.iter().filter(|i| !i.tool_calls.is_empty()).count(),
            MemoryIntegration::dedup_tags(interactions),
        )
    }

    fn extract_tags(summary: &str) -> Vec<String> {
        summary.split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .collect()
    }

    async fn store(
        &self,
        summary: String,
        tags: Vec<String>,
    ) -> Result<(), MemoryError> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let tokens = estimate_tokens(&summary);

        sqlx::query(
            r#"
            INSERT INTO compressed_memories
                (id, level, summary, tags, source_count, token_estimate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(id.to_string())
        .bind(1 as i64)
        .bind(summary)
        .bind(serde_json::to_string(&tags).map_err(|e| MemoryError::Db(e.to_string()))?)
        .bind(0 as i64)
        .bind(tokens as i64)
        .bind(now.to_rfc3339())
        .execute(&self.recall_pool)
        .await
        .map_err(|e| MemoryError::Db(e.to_string()))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Rough token estimate: 1 token per 4 characters of text.
fn estimate_tokens(text: &str) -> u32 {
    (text.len() / 4).max(1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_interval_hours() {
        assert_eq!(MemoryIntegration::parse_interval("6h"), Some(Duration::from_secs(6 * 3600)));
    }

    #[test]
    fn parse_interval_minutes() {
        assert_eq!(MemoryIntegration::parse_interval("30m"), Some(Duration::from_secs(30 * 60)));
    }

    #[test]
    fn parse_interval_seconds() {
        assert_eq!(MemoryIntegration::parse_interval("90s"), Some(Duration::from_secs(90)));
    }

    #[test]
    fn parse_interval_days() {
        assert_eq!(
            MemoryIntegration::parse_interval("2d"),
            Some(Duration::from_secs(2 * 86400))
        );
    }

    #[test]
    fn parse_interval_invalid() {
        assert!(MemoryIntegration::parse_interval("abc").is_none());
        assert!(MemoryIntegration::parse_interval("").is_none());
    }

    #[test]
    fn estimate_tokens_basic() {
        assert!(estimate_tokens("hello") >= 1);
        assert!(estimate_tokens("this is a longer piece of text to estimate") > estimate_tokens("hi"));
    }

    #[test]
    fn tags_overlap_matches() {
        let stored = vec!["rust".into(), "memory".into()];
        let query = vec!["memory".into()];
        assert!(MemoryIntegration::tags_overlap(&stored, &query));
    }

    #[test]
    fn tags_overlap_no_match() {
        let stored = vec!["rust".into()];
        let query = vec!["python".into()];
        assert!(!MemoryIntegration::tags_overlap(&stored, &query));
    }

    #[test]
    fn tags_overlap_empty() {
        let stored: Vec<String> = vec![];
        let query = vec!["x".into()];
        assert!(!MemoryIntegration::tags_overlap(&stored, &query));
    }
}
