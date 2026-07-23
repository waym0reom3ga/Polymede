use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::process::Command;

// ---------------------------------------------------------------------------
// Public types (backward compatible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub output: String,
    pub ok: bool,
}

// ---------------------------------------------------------------------------
// LRU cache entry for tool results
// ---------------------------------------------------------------------------

struct CacheEntry {
    result: ToolResult,
    inserted_at: Instant,
}

/// Simple bounded LRU cache keyed by (tool_name, canonical_args_json).
#[derive(Clone)]
pub struct ToolCache {
    inner: Arc<tokio::sync::RwLock<ToolCacheInner>>,
    ttl: Duration,
    max_entries: usize,
}

struct ToolCacheInner {
    map: HashMap<String, CacheEntry>,
    order: VecDeque<String>, // LRU order (front = least recent)
}

impl ToolCache {
    pub fn new(ttl_secs: u64, max_entries: usize) -> Self {
        Self {
            inner: Arc::new(tokio::sync::RwLock::new(ToolCacheInner {
                map: HashMap::new(),
                order: VecDeque::with_capacity(max_entries),
            })),
            ttl: Duration::from_secs(ttl_secs),
            max_entries,
        }
    }

    /// Build a cache key from tool name + canonical JSON args.
    fn key(name: &str, args: &serde_json::Value) -> String {
        format!("{}:{}", name, serde_json::to_string(args).unwrap_or_default())
    }

    pub async fn get(&self, name: &str, args: &serde_json::Value) -> Option<ToolResult> {
        let k = Self::key(name, args);
        let mut inner = self.inner.write().await;

        // Check if entry exists and is fresh.
        let (hit, expired) = if let Some(entry) = inner.map.get(&k) {
            if entry.inserted_at.elapsed() < self.ttl {
                (Some(entry.result.clone()), false)
            } else {
                (None, true)
            }
        } else {
            (None, false)
        };

        // Update LRU order or evict expired — no longer holding entry ref.
        if hit.is_some() {
            inner.order.retain(|x| x != &k);
            inner.order.push_back(k);
        } else if expired {
            inner.map.remove(&k);
            inner.order.retain(|x| x != &k);
        }

        hit
    }

    pub async fn put(&self, name: &str, args: &serde_json::Value, result: ToolResult) {
        let k = Self::key(name, args);
        let mut inner = self.inner.write().await;
        // Evict LRU if full
        while inner.map.len() >= self.max_entries && !inner.order.is_empty() {
            if let Some(lru_key) = inner.order.pop_front() {
                inner.map.remove(&lru_key);
            }
        }
        inner.map.insert(
            k.clone(),
            CacheEntry {
                result,
                inserted_at: Instant::now(),
            },
        );
        inner.order.push_back(k);
    }

    pub async fn clear(&self) {
        let mut inner = self.inner.write().await;
        inner.map.clear();
        inner.order.clear();
    }
}

// ---------------------------------------------------------------------------
// Shared context passed to every tool
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ToolContext {
    pub working_dir: PathBuf,
    pub allowed_commands: Option<Vec<String>>,
}

impl ToolContext {
    pub fn new(working_dir: PathBuf, allowed_commands: Option<Vec<String>>) -> Self {
        Self {
            working_dir,
            allowed_commands,
        }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        if path.starts_with('/') {
            PathBuf::from(path)
        } else {
            self.working_dir.join(path)
        }
    }
}

// ---------------------------------------------------------------------------
// Tool trait
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;
    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult;
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
    index: HashMap<String, Arc<dyn Tool>>,
    cache: Option<ToolCache>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::with_tools(Self::default_tools())
    }

    /// Enable result caching with TTL (seconds) and max entries.
    pub fn with_cache(mut self, ttl_secs: u64, max_entries: usize) -> Self {
        self.cache = Some(ToolCache::new(ttl_secs, max_entries));
        self
    }

    pub fn with_tools(tools: Vec<Arc<dyn Tool>>) -> Self {
        let index = tools
            .iter()
            .map(|t| (t.name().to_string(), Arc::clone(t)))
            .collect();
        Self {
            tools,
            index,
            cache: None,
        }
    }

    pub fn definitions(&self) -> Vec<serde_json::Value> {
        self.tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.parameters()
                    }
                })
            })
            .collect()
    }

    pub async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> ToolResult {
        // Try cache first.
        if let Some(ref cache) = self.cache {
            if let Some(hit) = cache.get(&call.name, &call.arguments).await {
                tracing::debug!(tool = %call.name, "cache hit");
                return hit;
            }
        }

        let result = if let Some(tool) = self.index.get(&call.name) {
            tracing::debug!(tool = %call.name, "executing tool");
            tool.execute(ctx, &call.arguments).await
        } else {
            tracing::warn!(tool = %call.name, "unknown tool");
            ToolResult {
                name: call.name.clone(),
                output: format!("Unknown tool: {}", call.name),
                ok: false,
            }
        };

        // Store in cache.
        if let Some(ref cache) = self.cache {
            cache.put(&call.name, &call.arguments, result.clone()).await;
        }

        result
    }

    /// Clear the tool result cache (called on /reset).
    pub async fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.clear().await;
        }
    }

    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    pub fn default_tools() -> Vec<Arc<dyn Tool>> {
        vec![
            // Terminal
            Arc::new(BashTool),
            Arc::new(EnvTool),
            Arc::new(WhichTool),
            Arc::new(PingTool),
            Arc::new(CurlTool),
            Arc::new(DigTool),
            // File operations
            Arc::new(ReadTool),
            Arc::new(WriteTool),
            Arc::new(EditTool),
            Arc::new(MkdirTool),
            Arc::new(RmTool),
            Arc::new(CpTool),
            Arc::new(MvTool),
            Arc::new(TouchTool),
            Arc::new(ChmodTool),
            Arc::new(Base64Tool),
            // Search
            Arc::new(GrepTool),
            Arc::new(GlobTool),
            Arc::new(FindTool),
            Arc::new(WhereisTool),
            // Web
            Arc::new(WebSearchTool),
            Arc::new(WebFetchTool),
            Arc::new(UrlInfoTool),
            Arc::new(DnsLookupTool),
            // AI / media
            Arc::new(ImageGenerateTool),
            Arc::new(TextToSpeechTool),
            Arc::new(SpeechToTextTool),
            Arc::new(TranslateTool),
            Arc::new(SummarizeTool),
            // System
            Arc::new(DateTool),
            Arc::new(UptimeTool),
            Arc::new(DiskUsageTool),
            Arc::new(ProcessListTool),
            Arc::new(NetworkInfoTool),
            // Developer
            Arc::new(GitStatusTool),
            Arc::new(GitLogTool),
            Arc::new(GitDiffTool),
            Arc::new(CompilerCheckTool),
            Arc::new(PackageSearchTool),
            // Interaction
            Arc::new(QuestionTool),
            Arc::new(NotifyTool),
        ]
    }
}

// ---------------------------------------------------------------------------
// Helper: resolve string from JSON args
// ---------------------------------------------------------------------------

fn arg_str(args: &serde_json::Value, key: &str) -> Option<String> {
    args.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
}

fn arg_str_required(args: &serde_json::Value, key: &str) -> Result<String, String> {
    arg_str(args, key).ok_or_else(|| format!("missing required argument: {}", key))
}

fn arg_bool(args: &serde_json::Value, key: &str) -> bool {
    args.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

fn arg_i64(args: &serde_json::Value, key: &str) -> Option<i64> {
    args.get(key).and_then(|v| v.as_i64())
}

fn arg_u64(args: &serde_json::Value, key: &str) -> Option<u64> {
    args.get(key).and_then(|v| v.as_u64())
}

fn ok_result(name: &str, output: String) -> ToolResult {
    ToolResult {
        name: name.into(),
        output,
        ok: true,
    }
}

fn err_result(name: &str, output: String) -> ToolResult {
    ToolResult {
        name: name.into(),
        output,
        ok: false,
    }
}

// ---------------------------------------------------------------------------
// Terminal tools
// ---------------------------------------------------------------------------

struct BashTool;

#[async_trait::async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command with timeout support. Runs in the working directory."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in milliseconds, default 30000"}
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let command = match arg_str_required(args, "command") {
            Ok(c) => c,
            Err(e) => return err_result("bash", e),
        };

        if command.is_empty() {
            return err_result("bash", "Error: empty command".into());
        }

        let timeout_ms: u64 = arg_u64(args, "timeout").unwrap_or(30000);

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(&command).current_dir(&ctx.working_dir);

        if let Some(ref allowed) = ctx.allowed_commands {
            let base = command.split_whitespace().next().unwrap_or("");
            if !allowed.iter().any(|a| a.eq_ignore_ascii_case(base)) {
                return err_result(
                    "bash",
                    format!("Command '{}' is not in the allowed list", base),
                );
            }
        }

        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn: {e}"));

        let child = match output {
            Ok(c) => c,
            Err(e) => return err_result("bash", e),
        };

        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_millis(timeout_ms);

        match tokio::time::timeout_at(deadline, child.wait_with_output()).await {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let combined = if stderr.is_empty() {
                    stdout
                } else {
                    format!("{stdout}\nstderr:\n{stderr}")
                };
                ToolResult {
                    name: "bash".into(),
                    output: combined,
                    ok: output.status.success(),
                }
            }
            Ok(Err(e)) => err_result("bash", format!("Execution error: {e}")),
            Err(_) => err_result("bash", format!("Command timed out after {timeout_ms}ms")),
        }
    }
}

struct EnvTool;

#[async_trait::async_trait]
impl Tool for EnvTool {
    fn name(&self) -> &str {
        "env"
    }

    fn description(&self) -> &str {
        "Get or set environment variables."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "'get', 'list', or 'set'"},
                "key": {"type": "string", "description": "Variable name"},
                "value": {"type": "string", "description": "Variable value (for set)"}
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let action = arg_str(args, "action").unwrap_or_else(|| "list".into());

        match action.as_str() {
            "get" => {
                let key = match arg_str_required(args, "key") {
                    Ok(k) => k,
                    Err(e) => return err_result("env", e),
                };
                let val = std::env::var(&key);
                match val {
                    Ok(v) => ok_result("env", format!("{}={}", key, v)),
                    Err(_) => err_result("env", format!("Variable '{}' not found", key)),
                }
            }
            "set" => {
                let key = match arg_str_required(args, "key") {
                    Ok(k) => k,
                    Err(e) => return err_result("env", e),
                };
                let value = match arg_str_required(args, "value") {
                    Ok(v) => v,
                    Err(e) => return err_result("env", e),
                };
                unsafe { std::env::set_var(&key, &value) };
                ok_result("env", format!("Set {}={}", key, value))
            }
            "list" => {
                let vars: String = std::env::vars()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join("\n");
                ok_result("env", vars)
            }
            _ => err_result("env", format!("Unknown action: {}", action)),
        }
    }
}

struct WhichTool;

#[async_trait::async_trait]
impl Tool for WhichTool {
    fn name(&self) -> &str {
        "which"
    }

    fn description(&self) -> &str {
        "Find the full path of an executable."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "Program name to locate"}
            },
            "required": ["program"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let program = match arg_str_required(args, "program") {
            Ok(p) => p,
            Err(e) => return err_result("which", e),
        };
        let result = BashTool.execute(
            ctx,
            &serde_json::json!({"command": format!("which {}", program)}),
        )
        .await;
        ToolResult {
            name: "which".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct PingTool;

#[async_trait::async_trait]
impl Tool for PingTool {
    fn name(&self) -> &str {
        "ping"
    }

    fn description(&self) -> &str {
        "Ping a host to check connectivity."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Hostname or IP address"},
                "count": {"type": "integer", "description": "Number of pings, default 4"}
            },
            "required": ["host"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let host = match arg_str_required(args, "host") {
            Ok(h) => h,
            Err(e) => return err_result("ping", e),
        };
        let count: i64 = arg_i64(args, "count").unwrap_or(4);
        let result = BashTool.execute(
            ctx,
            &serde_json::json!({
                "command": format!("ping -c {} {}", count, host),
                "timeout": 15000
            }),
        )
        .await;
        ToolResult {
            name: "ping".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct CurlTool;

#[async_trait::async_trait]
impl Tool for CurlTool {
    fn name(&self) -> &str {
        "curl"
    }

    fn description(&self) -> &str {
        "Make an HTTP request and return the response body."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to request"},
                "method": {"type": "string", "description": "HTTP method, default GET"},
                "headers": {"type": "object", "description": "Additional headers as key-value pairs"},
                "body": {"type": "string", "description": "Request body for POST/PUT"}
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let url = match arg_str_required(args, "url") {
            Ok(u) => u,
            Err(e) => return err_result("curl", e),
        };

        let method: String = arg_str(args, "method").unwrap_or_else(|| "GET".into());

        let mut builder = match method.as_str() {
            "GET" => reqwest::Client::new().get(&url),
            "POST" => reqwest::Client::new().post(&url),
            "PUT" => reqwest::Client::new().put(&url),
            "DELETE" => reqwest::Client::new().delete(&url),
            "PATCH" => reqwest::Client::new().patch(&url),
            _ => return err_result("curl", format!("Unsupported method: {}", method)),
        };

        if let Some(body) = arg_str(args, "body") {
            builder = builder.body(body);
        }

        if let Some(headers) = args.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in headers {
                if let Some(val) = v.as_str() {
                    builder = builder.header(k, val);
                }
            }
        }

        match builder.send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();
                ok_result("curl", format!("HTTP {}\n{}", status, body))
            }
            Err(e) => err_result("curl", format!("Request failed: {}", e)),
        }
    }
}

struct DigTool;

#[async_trait::async_trait]
impl Tool for DigTool {
    fn name(&self) -> &str {
        "dig"
    }

    fn description(&self) -> &str {
        "Perform DNS lookup for a domain."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Domain to look up"},
                "record_type": {"type": "string", "description": "DNS record type, default A"}
            },
            "required": ["domain"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let domain = match arg_str_required(args, "domain") {
            Ok(d) => d,
            Err(e) => return err_result("dig", e),
        };
        let rtype: String = arg_str(args, "record_type").unwrap_or_else(|| "A".into());
        let result = BashTool.execute(
            ctx,
            &serde_json::json!({
                "command": format!("dig +short {} {}", domain, rtype),
                "timeout": 10000
            }),
        )
        .await;
        ToolResult {
            name: "dig".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

// ---------------------------------------------------------------------------
// File operation tools
// ---------------------------------------------------------------------------

struct ReadTool;

#[async_trait::async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read a file or list a directory. Supports offset and limit for large files."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path"},
                "offset": {"type": "integer", "description": "Line number to start from (1-indexed)"},
                "limit": {"type": "integer", "description": "Max lines to read, default 200"}
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let path = match arg_str_required(args, "path") {
            Ok(p) => p,
            Err(e) => return err_result("read", e),
        };

        let full_path = ctx.resolve_path(&path);

        if !full_path.exists() {
            return err_result(
                "read",
                format!("Path does not exist: {}", full_path.display()),
            );
        }

        if full_path.is_dir() {
            let entries: Vec<String> = match fs::read_dir(&full_path).await {
                Ok(mut rd) => {
                    let mut items = Vec::new();
                    while let Ok(Some(entry)) = rd.next_entry().await {
                        let name = entry.file_name();
                        let name_str = name.to_string_lossy().to_string();
                        let ft = entry.file_type().await;
                        let suffix = if ft.map_or(false, |t| t.is_dir()) {
                            "/"
                        } else {
                            ""
                        };
                        items.push(format!("{name_str}{suffix}"));
                    }
                    items
                }
                Err(e) => {
                    return err_result("read", format!("Failed to read directory: {e}"));
                }
            };
            ok_result("read", entries.join("\n"))
        } else {
            let content = match fs::read_to_string(&full_path).await {
                Ok(c) => c,
                Err(e) => {
                    return err_result("read", format!("Failed to read file: {e}"));
                }
            };

            let offset: usize = arg_u64(args, "offset").map(|v| v as usize).unwrap_or(1);
            let limit: usize = arg_u64(args, "limit").map(|v| v as usize).unwrap_or(200);

            let lines: Vec<&str> = content.lines().collect();
            let start = if offset > lines.len() {
                lines.len()
            } else {
                offset - 1
            };
            let end = (start + limit).min(lines.len());

            let mut output = String::new();
            for (i, line) in lines[start..end].iter().enumerate() {
                output.push_str(&format!("{}: {}\n", start + i + 1, line));
            }

            if lines.len() > limit {
                output.push_str(&format!(
                    "\n...(truncated, {} total lines, show more with offset={})\n",
                    lines.len(),
                    end + 1
                ));
            }

            ok_result("read", output)
        }
    }
}

struct WriteTool;

#[async_trait::async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates parent directories as needed."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "File content"}
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let path = match arg_str_required(args, "path") {
            Ok(p) => p,
            Err(e) => return err_result("write", e),
        };
        let content = match arg_str_required(args, "content") {
            Ok(c) => c,
            Err(e) => return err_result("write", e),
        };

        let full_path = ctx.resolve_path(&path);

        if let Some(parent) = full_path.parent() {
            if let Err(e) = fs::create_dir_all(parent).await {
                return err_result("write", format!("Failed to create directory: {e}"));
            }
        }

        match fs::write(&full_path, &content).await {
            Ok(()) => ok_result(
                "write",
                format!("Wrote {} bytes to {}", content.len(), full_path.display()),
            ),
            Err(e) => err_result("write", format!("Failed to write file: {e}")),
        }
    }
}

struct EditTool;

#[async_trait::async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Perform in-place text replacement in a file. Supports single or multiple replacements."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "old_text": {"type": "string", "description": "Text to find and replace"},
                "new_text": {"type": "string", "description": "Replacement text"},
                "global": {"type": "boolean", "description": "Replace all occurrences, default true"}
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let path = match arg_str_required(args, "path") {
            Ok(p) => p,
            Err(e) => return err_result("edit", e),
        };
        let old_text = match arg_str_required(args, "old_text") {
            Ok(o) => o,
            Err(e) => return err_result("edit", e),
        };
        let new_text = match arg_str_required(args, "new_text") {
            Ok(n) => n,
            Err(e) => return err_result("edit", e),
        };
        let global = arg_bool(args, "global") || true;

        let full_path = ctx.resolve_path(&path);

        let content = match fs::read_to_string(&full_path).await {
            Ok(c) => c,
            Err(e) => return err_result("edit", format!("Failed to read file: {e}")),
        };

        let new_content = if global {
            content.replace(&old_text, &new_text)
        } else {
            content.replacen(&old_text, &new_text, 1)
        };

        let count = if global {
            content.matches(&old_text).count()
        } else {
            if content.contains(&old_text) { 1 } else { 0 }
        };

        if count == 0 {
            return err_result(
                "edit",
                format!("Text not found in file: {}", full_path.display()),
            );
        }

        match fs::write(&full_path, &new_content).await {
            Ok(()) => ok_result(
                "edit",
                format!(
                    "Replaced {} occurrence(s) in {}",
                    count,
                    full_path.display()
                ),
            ),
            Err(e) => err_result("edit", format!("Failed to write file: {e}")),
        }
    }
}

struct MkdirTool;

#[async_trait::async_trait]
impl Tool for MkdirTool {
    fn name(&self) -> &str {
        "mkdir"
    }

    fn description(&self) -> &str {
        "Create a directory, including parents if recursive is true."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to create"},
                "recursive": {"type": "boolean", "description": "Create parent dirs, default true"}
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let path = match arg_str_required(args, "path") {
            Ok(p) => p,
            Err(e) => return err_result("mkdir", e),
        };
        let full_path = ctx.resolve_path(&path);

        match fs::create_dir_all(&full_path).await {
            Ok(()) => ok_result("mkdir", format!("Created: {}", full_path.display())),
            Err(e) => err_result("mkdir", format!("Failed to create directory: {e}")),
        }
    }
}

struct RmTool;

#[async_trait::async_trait]
impl Tool for RmTool {
    fn name(&self) -> &str {
        "rm"
    }

    fn description(&self) -> &str {
        "Remove a file or directory. Use recursive=true for directories."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to remove"},
                "recursive": {"type": "boolean", "description": "Remove recursively for dirs"}
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let path = match arg_str_required(args, "path") {
            Ok(p) => p,
            Err(e) => return err_result("rm", e),
        };
        let recursive = arg_bool(args, "recursive");
        let full_path = ctx.resolve_path(&path);

        if !full_path.exists() {
            return err_result("rm", format!("Path not found: {}", full_path.display()));
        }

        let result = if full_path.is_dir() && recursive {
            fs::remove_dir_all(&full_path).await
        } else if full_path.is_dir() {
            fs::remove_dir(&full_path).await
        } else {
            fs::remove_file(&full_path).await
        };

        match result {
            Ok(()) => ok_result("rm", format!("Removed: {}", full_path.display())),
            Err(e) => err_result("rm", format!("Failed to remove: {e}")),
        }
    }
}

struct CpTool;

#[async_trait::async_trait]
impl Tool for CpTool {
    fn name(&self) -> &str {
        "cp"
    }

    fn description(&self) -> &str {
        "Copy a file or directory to a destination."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path"},
                "destination": {"type": "string", "description": "Destination path"}
            },
            "required": ["source", "destination"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let source = match arg_str_required(args, "source") {
            Ok(s) => s,
            Err(e) => return err_result("cp", e),
        };
        let destination = match arg_str_required(args, "destination") {
            Ok(d) => d,
            Err(e) => return err_result("cp", e),
        };

        let src = ctx.resolve_path(&source);
        let dst = ctx.resolve_path(&destination);

        if !src.exists() {
            return err_result("cp", format!("Source not found: {}", src.display()));
        }

        match fs::copy(&src, &dst).await {
            Ok(n) => ok_result("cp", format!("Copied {} bytes to {}", n, dst.display())),
            Err(e) => err_result("cp", format!("Copy failed: {e}")),
        }
    }
}

struct MvTool;

#[async_trait::async_trait]
impl Tool for MvTool {
    fn name(&self) -> &str {
        "mv"
    }

    fn description(&self) -> &str {
        "Move or rename a file or directory."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path"},
                "destination": {"type": "string", "description": "Destination path"}
            },
            "required": ["source", "destination"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let source = match arg_str_required(args, "source") {
            Ok(s) => s,
            Err(e) => return err_result("mv", e),
        };
        let destination = match arg_str_required(args, "destination") {
            Ok(d) => d,
            Err(e) => return err_result("mv", e),
        };

        let src = ctx.resolve_path(&source);
        let dst = ctx.resolve_path(&destination);

        if !src.exists() {
            return err_result("mv", format!("Source not found: {}", src.display()));
        }

        match fs::rename(&src, &dst).await {
            Ok(()) => ok_result("mv", format!("Moved to {}", dst.display())),
            Err(e) => err_result("mv", format!("Move failed: {e}")),
        }
    }
}

struct TouchTool;

#[async_trait::async_trait]
impl Tool for TouchTool {
    fn name(&self) -> &str {
        "touch"
    }

    fn description(&self) -> &str {
        "Create an empty file or update its timestamp."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let path = match arg_str_required(args, "path") {
            Ok(p) => p,
            Err(e) => return err_result("touch", e),
        };
        let full_path = ctx.resolve_path(&path);

        match fs::File::create(&full_path).await {
            Ok(_) => ok_result("touch", format!("Touched: {}", full_path.display())),
            Err(e) => err_result("touch", format!("Failed: {e}")),
        }
    }
}

struct ChmodTool;

#[async_trait::async_trait]
impl Tool for ChmodTool {
    fn name(&self) -> &str {
        "chmod"
    }

    fn description(&self) -> &str {
        "Change file permissions using octal mode."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path"},
                "mode": {"type": "string", "description": "Octal permission mode, e.g. 755"}
            },
            "required": ["path", "mode"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let path = match arg_str_required(args, "path") {
            Ok(p) => p,
            Err(e) => return err_result("chmod", e),
        };
        let mode = match arg_str_required(args, "mode") {
            Ok(m) => m,
            Err(e) => return err_result("chmod", e),
        };

        let full_path = ctx.resolve_path(&path);

        let result = BashTool.execute(
            ctx,
            &serde_json::json!({
                "command": format!("chmod {} {}", mode, full_path.display())
            }),
        )
        .await;

        ToolResult {
            name: "chmod".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct Base64Tool;

#[async_trait::async_trait]
impl Tool for Base64Tool {
    fn name(&self) -> &str {
        "base64"
    }

    fn description(&self) -> &str {
        "Encode or decode base64 content."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "'encode' or 'decode'"},
                "content": {"type": "string", "description": "Content to encode or decode"}
            },
            "required": ["action", "content"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let action = match arg_str_required(args, "action") {
            Ok(a) => a,
            Err(e) => return err_result("base64", e),
        };
        let content = match arg_str_required(args, "content") {
            Ok(c) => c,
            Err(e) => return err_result("base64", e),
        };

        match action.as_str() {
            "encode" => {
                let encoded = base64_encode(&content);
                ok_result("base64", encoded)
            }
            "decode" => match base64_decode(&content) {
                Ok(decoded) => ok_result("base64", decoded),
                Err(e) => err_result("base64", format!("Decode failed: {e}")),
            },
            _ => err_result("base64", format!("Unknown action: {}", action)),
        }
    }
}

fn base64_encode(input: &str) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(input.as_bytes())
}

fn base64_decode(input: &str) -> Result<String, String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(input)
        .map_err(|e| format!("invalid base64: {e}"))?;
    String::from_utf8(bytes).map_err(|e| format!("invalid utf-8: {e}"))
}

// ---------------------------------------------------------------------------
// Search tools
// ---------------------------------------------------------------------------

struct GrepTool;

#[async_trait::async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents with regex. Supports include filter for file extensions."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search in"},
                "include": {"type": "string", "description": "File pattern to include, e.g. '*.rs'"}
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let pattern = match arg_str_required(args, "pattern") {
            Ok(p) => p,
            Err(e) => return err_result("grep", e),
        };

        let search_path = arg_str(args, "path")
            .map(|p| ctx.resolve_path(&p))
            .unwrap_or_else(|| ctx.working_dir.clone());

        let include = arg_str(args, "include");

        let grep_cmd = if let Some(pattern_filter) = include {
            format!(
                "grep -rnE '{}' --include='{}' {}",
                pattern,
                pattern_filter,
                search_path.display()
            )
        } else {
            format!("grep -rnE '{}' {}", pattern, search_path.display())
        };

        let output = Command::new("sh")
            .arg("-c")
            .arg(&grep_cmd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let output = match output {
            Ok(c) => c.wait_with_output().await.map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        };

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                let result = if stdout.is_empty() && !stderr.is_empty() {
                    format!("No matches found\n{stderr}")
                } else {
                    stdout
                };
                ok_result("grep", result)
            }
            Err(e) => err_result("grep", format!("grep error: {e}")),
        }
    }
}

struct GlobTool;

#[async_trait::async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files by glob pattern. E.g. 'src/**/*.rs'."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match"},
                "path": {"type": "string", "description": "Base directory for search"}
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let pattern = match arg_str_required(args, "pattern") {
            Ok(p) => p,
            Err(e) => return err_result("glob", e),
        };

        let search_path = arg_str(args, "path")
            .map(|p| ctx.resolve_path(&p))
            .unwrap_or_else(|| ctx.working_dir.clone());

        let full_pattern = search_path.join(&pattern);
        let pattern_str = full_pattern.to_str().unwrap_or("");

        let entries = match glob::glob(pattern_str) {
            Ok(e) => e,
            Err(e) => return err_result("glob", format!("Invalid glob pattern: {e}")),
        };

        let mut results = Vec::new();
        for entry in entries {
            match entry {
                Ok(path) => results.push(path.display().to_string()),
                Err(e) => results.push(format!("Error: {e}")),
            }
        }

        if results.is_empty() {
            ok_result("glob", "No matches found".into())
        } else {
            ok_result("glob", results.join("\n"))
        }
    }
}

struct FindTool;

#[async_trait::async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files or directories by name, type, or other criteria using the find command."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to search"},
                "name": {"type": "string", "description": "Name pattern to match"},
                "type": {"type": "string", "description": "Type filter: 'f' for files, 'd' for dirs"},
                "max_depth": {"type": "integer", "description": "Maximum recursion depth"}
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let search_path = arg_str(args, "path")
            .map(|p| ctx.resolve_path(&p))
            .unwrap_or_else(|| ctx.working_dir.clone());

        let name = arg_str(args, "name");
        let ftype = arg_str(args, "type");
        let max_depth = arg_i64(args, "max_depth");

        let mut cmd_parts = vec![format!("find '{}'", search_path.display())];

        if let Some(depth) = max_depth {
            cmd_parts.push(format!("-maxdepth {}", depth));
        }

        if let Some(t) = ftype {
            cmd_parts.push(format!("-type {}", t));
        }

        if let Some(n) = name {
            cmd_parts.push(format!("-name '{}'", n));
        }

        cmd_parts.push("2>/dev/null".into());
        cmd_parts.push("| head -100".into());

        let result = BashTool.execute(
            ctx,
            &serde_json::json!({"command": cmd_parts.join(" ")}),
        )
        .await;

        ToolResult {
            name: "find".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct WhereisTool;

#[async_trait::async_trait]
impl Tool for WhereisTool {
    fn name(&self) -> &str {
        "whereis"
    }

    fn description(&self) -> &str {
        "Locate binary, source, and manual page files for a command."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command name to locate"}
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let command = match arg_str_required(args, "command") {
            Ok(c) => c,
            Err(e) => return err_result("whereis", e),
        };
        let result = BashTool.execute(
            ctx,
            &serde_json::json!({"command": format!("whereis {}", command)}),
        )
        .await;
        ToolResult {
            name: "whereis".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

// ---------------------------------------------------------------------------
// Web tools
// ---------------------------------------------------------------------------

struct WebSearchTool;

#[async_trait::async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for information. Returns titles, URLs, and snippets."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results, default 5"}
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let query = match arg_str_required(args, "query") {
            Ok(q) => q,
            Err(e) => return err_result("web_search", e),
        };

        let num_results: usize = arg_u64(args, "num_results")
            .map(|v| v as usize)
            .unwrap_or(5);

        // Use DuckDuckGo HTML search as a free, no-API-key option
        let url = format!(
            "https://html.duckduckgo.com/html/?q={}",
            urlencoding(&query)
        );

        match reqwest::Client::new()
            .get(&url)
            .header("User-Agent", "Polymede/1.0")
            .send()
            .await
        {
            Ok(resp) => {
                let body = match resp.text().await {
                    Ok(b) => b,
                    Err(e) => return err_result("web_search", format!("Failed to read response: {e}")),
                };

                let results = parse_ddg_results(&body, num_results);
                if results.is_empty() {
                    ok_result("web_search", "No results found".into())
                } else {
                    let output: Vec<String> = results
                        .iter()
                        .enumerate()
                        .map(|(i, r)| format!("[{}] {}\n    {}\n    {}", i + 1, r.title, r.url, r.snippet))
                        .collect();
                    ok_result("web_search", output.join("\n\n"))
                }
            }
            Err(e) => err_result("web_search", format!("Search failed: {e}")),
        }
    }
}

struct DdgResult {
    title: String,
    url: String,
    snippet: String,
}

fn parse_ddg_results(html: &str, max: usize) -> Vec<DdgResult> {
    let mut results = Vec::new();
    let lines: Vec<&str> = html.lines().collect();

    for i in 0..lines.len() {
        if results.len() >= max {
            break;
        }

        let line = lines[i];
        if line.contains("class=\"result__a\"") {
            let title = extract_attr(line, "data-result-title")
                .or_else(|| extract_text_between(line, ">", "</a>"))
                .unwrap_or_else(|| "No title".into());

            let url = extract_attr(line, "href")
                .and_then(|h| extract_ddg_redirect_url(&h))
                .unwrap_or_else(|| "No URL".into());

            let snippet = if i + 1 < lines.len() {
                extract_text_between(lines[i + 1], "class=\"result__snippet\"", "</a>")
                    .unwrap_or_default()
            } else {
                String::new()
            };

            results.push(DdgResult {
                title,
                url,
                snippet,
            });
        }
    }

    results
}

fn extract_attr(html: &str, attr: &str) -> Option<String> {
    let prefix = format!("{}=\"", attr);
    let start = html.find(&prefix)? + prefix.len();
    let rest = &html[start..];
    let end = rest.find('?')?;
    Some(rest[..end].to_string())
}

fn extract_text_between(html: &str, open: &str, close: &str) -> Option<String> {
    let start = html.find(open)? + open.len();
    let rest = &html[start..];
    let end = rest.find(close)?;
    Some(rest[..end].trim().to_string())
}

fn extract_ddg_redirect_url(href: &str) -> Option<String> {
    if href.starts_with("https://duckduckgo.com/l/?uddg=") {
        let encoded = &href["https://duckduckgo.com/l/?uddg=".len()..];
        urldecode(encoded)
    } else {
        None
    }
}

fn urlencoding(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.len_utf8() == 1 && is_unreserved(c) {
                c.to_string()
            } else {
                let mut out = String::new();
                for b in c.to_string().as_bytes() {
                    out.push_str(&format!("{:02X}", b));
                }
                out
            }
        })
        .collect()
}

fn is_unreserved(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~')
}

fn urldecode(s: &str) -> Option<String> {
    let mut result = Vec::new();
    let mut chars = s.bytes();
    while let Some(b) = chars.next() {
        if b == b'%' {
            let h1 = chars.next()?;
            let h2 = chars.next()?;
            let hex = format!("{}{}", h1 as char, h2 as char);
            let byte = u8::from_str_radix(&hex, 16).ok()?;
            result.push(byte);
        } else if b == b'+' {
            result.push(b' ');
        } else {
            result.push(b);
        }
    }
    String::from_utf8(result).ok()
}

struct WebFetchTool;

#[async_trait::async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch content from a URL and return it as text or markdown."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "format": {"type": "string", "description": "Response format: 'text' or 'html', default 'text'"},
                "timeout": {"type": "integer", "description": "Timeout in seconds, default 30"}
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let url = match arg_str_required(args, "url") {
            Ok(u) => u,
            Err(e) => return err_result("web_fetch", e),
        };

        let timeout: u64 = arg_u64(args, "timeout").unwrap_or(30);

        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(timeout);

        let resp = match tokio::time::timeout_at(
            deadline,
            reqwest::Client::new()
                .get(&url)
                .header("User-Agent", "Polymede/1.0")
                .send(),
        )
        .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => return err_result("web_fetch", format!("Request failed: {e}")),
            Err(_) => return err_result("web_fetch", format!("Request timed out after {}s", timeout)),
        };

        let body = match resp.text().await {
            Ok(b) => b,
            Err(e) => return err_result("web_fetch", format!("Failed to read body: {e}")),
        };

        let truncated = if body.len() > 10000 {
            format!(
                "{}\n\n... (truncated, {} total chars)",
                &body[..10000],
                body.len()
            )
        } else {
            body
        };

        ok_result("web_fetch", truncated)
    }
}

struct UrlInfoTool;

#[async_trait::async_trait]
impl Tool for UrlInfoTool {
    fn name(&self) -> &str {
        "url_info"
    }

    fn description(&self) -> &str {
        "Get metadata about a URL without downloading the full content."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to inspect"}
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let url = match arg_str_required(args, "url") {
            Ok(u) => u,
            Err(e) => return err_result("url_info", e),
        };

        match reqwest::Client::new().head(&url).send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let content_type = resp
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("unknown");
                let content_length = resp
                    .headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("unknown");
                let output = format!(
                    "Status: {}\nContent-Type: {}\nContent-Length: {}",
                    status, content_type, content_length
                );
                ok_result("url_info", output)
            }
            Err(e) => err_result("url_info", format!("Request failed: {e}")),
        }
    }
}

struct DnsLookupTool;

#[async_trait::async_trait]
impl Tool for DnsLookupTool {
    fn name(&self) -> &str {
        "dns_lookup"
    }

    fn description(&self) -> &str {
        "Resolve a domain name to IP addresses."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Domain name to resolve"}
            },
            "required": ["domain"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let domain = match arg_str_required(args, "domain") {
            Ok(d) => d,
            Err(e) => return err_result("dns_lookup", e),
        };

        match tokio::net::lookup_host(format!("{}:0", domain)).await {
            Ok(addrs) => {
                let ips: Vec<String> = addrs.map(|s| s.ip().to_string()).collect();
                ok_result(
                    "dns_lookup",
                    format!("{} resolves to:\n{}", domain, ips.join("\n")),
                )
            }
            Err(e) => err_result("dns_lookup", format!("Resolution failed: {e}")),
        }
    }
}

// ---------------------------------------------------------------------------
// AI / media tools
// ---------------------------------------------------------------------------

struct ImageGenerateTool;

#[async_trait::async_trait]
impl Tool for ImageGenerateTool {
    fn name(&self) -> &str {
        "image_generate"
    }

    fn description(&self) -> &str {
        "Generate an image from a text prompt using an AI image model."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text description of the image to generate"},
                "size": {"type": "string", "description": "Image size, default 1024x1024"},
                "model": {"type": "string", "description": "Model to use, default flux-schnell"}
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let prompt = match arg_str_required(args, "prompt") {
            Ok(p) => p,
            Err(e) => return err_result("image_generate", e),
        };

        let size: String = arg_str(args, "size").unwrap_or_else(|| "1024x1024".into());
        let model: String = arg_str(args, "model").unwrap_or_else(|| "flux-schnell".into());

        tracing::info!(prompt = %prompt, size = %size, model = %model, "image generation requested");

        // Try OpenAI-compatible image generation API (DALL-E / Flux via proxy)
        let api_key = std::env::var("POLYMDE_LLM_API_KEY").or_else(|_| std::env::var("OPENAI_API_KEY"));
        let base_url = std::env::var("POLYMDE_LLM_BASE_URL")
            .ok()
            .unwrap_or_else(|| "https://api.openai.com/v1".into());

        if let Ok(key) = api_key {
            match Self::call_image_api(&base_url, &key, &model, &prompt, &size).await {
                Ok(result) => return ok_result("image_generate", result),
                Err(e) => tracing::warn!(error = %e, "image API call failed, falling back"),
            }
        }

        // Fallback: local generation via bash (e.g., `python -c "import ..."` or `magick`)
        ok_result(
            "image_generate",
            format!(
                "Image generation requested for prompt: \"{}\" (model={}, size={}). \
                 No API key configured — set POLYMDE_LLM_API_KEY + POLYMDE_LLM_BASE_URL for real generation.",
                prompt, model, size
            ),
        )
    }
}

impl ImageGenerateTool {
    async fn call_image_api(
        base_url: &str,
        api_key: &str,
        model: &str,
        prompt: &str,
        size: &str,
    ) -> Result<String, String> {
        let url = format!("{}/images/generations", base_url.trim_end_matches('/'));
        let resp = reqwest::Client::new()
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&serde_json::json!({
                "model": model,
                "prompt": prompt,
                "size": size,
                "n": 1,
            }))
            .send()
            .await
            .map_err(|e| format!("request failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("API returned {}", resp.status()));
        }

        let body: serde_json::Value = resp.json().await.map_err(|e| format!("parse error: {}", e))?;

        // Handle both OpenAI-style (data[].url) and generic (images[]) responses
        if let Some(url) = body.get("data").and_then(|d| d.get(0)).and_then(|i| i.get("url")).and_then(|u| u.as_str()) {
            return Ok(format!("Image generated: {}", url));
        }
        if let Some(b64) = body.get("data").and_then(|d| d.get(0)).and_then(|i| i.get("b64_json")).and_then(|b| b.as_str()) {
            return Ok(format!(
                "Image generated (base64, {} chars). Set POLYMDE_IMAGE_SAVE_DIR to auto-save.",
                b64.len()
            ));
        }

        Err("Unexpected API response format".into())
    }
}

struct TextToSpeechTool;

#[async_trait::async_trait]
impl Tool for TextToSpeechTool {
    fn name(&self) -> &str {
        "text_to_speech"
    }

    fn description(&self) -> &str {
        "Convert text to speech audio."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to convert to speech"},
                "voice": {"type": "string", "description": "Voice name, default default"},
                "format": {"type": "string", "description": "Audio format, default mp3"}
            },
            "required": ["text"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let text = match arg_str_required(args, "text") {
            Ok(t) => t,
            Err(e) => return err_result("text_to_speech", e),
        };

        let voice: String = arg_str(args, "voice").unwrap_or_else(|| "alloy".into());
        let fmt: String = arg_str(args, "format").unwrap_or_else(|| "mp3".into());

        tracing::info!(text_len = text.len(), voice = %voice, format = %fmt, "TTS requested");

        // Try OpenAI-compatible TTS API (tts-1 / tts-1-hd)
        let api_key = std::env::var("POLYMDE_LLM_API_KEY").or_else(|_| std::env::var("OPENAI_API_KEY"));
        let base_url = std::env::var("POLYMDE_LLM_BASE_URL")
            .ok()
            .unwrap_or_else(|| "https://api.openai.com/v1".into());

        if let Ok(key) = api_key {
            match Self::call_tts_api(&base_url, &key, &voice, &text, &fmt).await {
                Ok(result) => return ok_result("text_to_speech", result),
                Err(e) => tracing::warn!(error = %e, "TTS API call failed, falling back"),
            }
        }

        ok_result(
            "text_to_speech",
            format!(
                "TTS requested: {} chars, voice '{}', format '{}'. \
                 No API key configured — set POLYMDE_LLM_API_KEY + POLYMDE_LLM_BASE_URL for real TTS.",
                text.len(), voice, fmt
            ),
        )
    }
}

impl TextToSpeechTool {
    async fn call_tts_api(
        base_url: &str,
        api_key: &str,
        voice: &str,
        text: &str,
        format: &str,
    ) -> Result<String, String> {
        let url = format!("{}/audio/speech", base_url.trim_end_matches('/'));
        let resp = reqwest::Client::new()
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&serde_json::json!({
                "model": if voice.ends_with("-hd") { "tts-1-hd" } else { "tts-1" },
                "voice": voice,
                "input": text,
                "response_format": format,
            }))
            .send()
            .await
            .map_err(|e| format!("request failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("API returned {}", resp.status()));
        }

        let bytes = resp.bytes().await.map_err(|e| format!("read error: {}", e))?;
        Ok(format!(
            "TTS audio generated ({} bytes, {} format). Set POLYMDE_AUDIO_SAVE_DIR to auto-save.",
            bytes.len(), format
        ))
    }
}

struct SpeechToTextTool;

#[async_trait::async_trait]
impl Tool for SpeechToTextTool {
    fn name(&self) -> &str {
        "speech_to_text"
    }

    fn description(&self) -> &str {
        "Transcribe audio to text."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "audio_path": {"type": "string", "description": "Path to the audio file"},
                "language": {"type": "string", "description": "Language code, default en"}
            },
            "required": ["audio_path"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let audio_path = match arg_str_required(args, "audio_path") {
            Ok(a) => a,
            Err(e) => return err_result("speech_to_text", e),
        };

        let full_path = ctx.resolve_path(&audio_path);

        if !full_path.exists() {
            return err_result(
                "speech_to_text",
                format!("Audio file not found: {}", full_path.display()),
            );
        }

        let language: String = arg_str(args, "language").unwrap_or_else(|| "en".into());

        tracing::info!(path = %full_path.display(), language = %language, "STT requested");

        // Try OpenAI-compatible Whisper API (whisper-1)
        let api_key = std::env::var("POLYMDE_LLM_API_KEY").or_else(|_| std::env::var("OPENAI_API_KEY"));
        let base_url = std::env::var("POLYMDE_LLM_BASE_URL")
            .ok()
            .unwrap_or_else(|| "https://api.openai.com/v1".into());

        if let Ok(key) = api_key {
            match Self::call_stt_api(&base_url, &key, &full_path, &language).await {
                Ok(result) => return ok_result("speech_to_text", result),
                Err(e) => tracing::warn!(error = %e, "STT API call failed, falling back"),
            }
        }

        // Fallback: try local whisper.cpp or ffmpeg + sox detection
        if let Ok(output) = Self::try_local_stt(&full_path).await {
            return ok_result("speech_to_text", output);
        }

        ok_result(
            "speech_to_text",
            format!(
                "STT requested for '{}' (language '{}'). \
                 No API key configured — set POLYMDE_LLM_API_KEY + POLYMDE_LLM_BASE_URL for real transcription.",
                full_path.display(), language
            ),
        )
    }
}

impl SpeechToTextTool {
    async fn call_stt_api(
        base_url: &str,
        api_key: &str,
        audio_path: &std::path::PathBuf,
        language: &str,
    ) -> Result<String, String> {
        let url = format!("{}/audio/transcriptions", base_url.trim_end_matches('/'));

        // Read file as multipart form data
        let file_bytes = tokio::fs::read(audio_path)
            .await
            .map_err(|e| format!("failed to read audio file: {}", e))?;

        let ext = audio_path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("mp3");

        let form = reqwest::multipart::Form::new()
            .part("file", reqwest::multipart::Part::bytes(file_bytes)
                .file_name(format!("audio.{}", ext)))
            .text("model", "whisper-1".to_string())
            .text("language", language.to_string());

        let resp = reqwest::Client::new()
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("request failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("API returned {}", resp.status()));
        }

        let body: serde_json::Value = resp.json().await.map_err(|e| format!("parse error: {}", e))?;
        if let Some(text) = body.get("text").and_then(|t| t.as_str()) {
            return Ok(format!("Transcription ({} chars):\n{}", text.len(), text));
        }

        Err("Unexpected API response format".into())
    }

    async fn try_local_stt(audio_path: &std::path::PathBuf) -> Result<String, String> {
        // Try whisper.cpp CLI if available
        let output = tokio::process::Command::new("whisper")
            .arg("-m")
            .arg("/usr/local/share/whisper/ggml-base.bin")
            .arg(audio_path)
            .output()
            .await;

        match output {
            Ok(o) if o.status.success() => {
                let text = String::from_utf8_lossy(&o.stdout);
                Ok(format!("Local transcription (whisper.cpp):\n{}", text.trim()))
            }
            _ => Err("local whisper not available".into()),
        }
    }
}

struct TranslateTool;

#[async_trait::async_trait]
impl Tool for TranslateTool {
    fn name(&self) -> &str {
        "translate"
    }

    fn description(&self) -> &str {
        "Translate text between languages."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "source_lang": {"type": "string", "description": "Source language code"},
                "target_lang": {"type": "string", "description": "Target language code, default en"}
            },
            "required": ["text", "target_lang"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let text = match arg_str_required(args, "text") {
            Ok(t) => t,
            Err(e) => return err_result("translate", e),
        };
        let target_lang = match arg_str_required(args, "target_lang") {
            Ok(t) => t,
            Err(e) => return err_result("translate", e),
        };
        let source_lang: String = arg_str(args, "source_lang").unwrap_or_else(|| "auto".into());

        tracing::info!(source = %source_lang, target = %target_lang, text_len = text.len(), "translation requested");

        // Try LibreTranslate / MyMemory API (free tier)
        match Self::call_translate_api(&text, &source_lang, &target_lang).await {
            Ok(result) => return ok_result("translate", result),
            Err(e) => tracing::warn!(error = %e, "translation API failed, falling back"),
        }

        // Fallback: use LLM if available (via bash calling polymede itself) or just inform user
        ok_result(
            "translate",
            format!(
                "Translation from '{}' to '{}' for {} chars. \
                 No translation service configured — set POLYMDE_TRANSLATE_API_URL for real translation.",
                source_lang, target_lang, text.len()
            ),
        )
    }
}

impl TranslateTool {
    async fn call_translate_api(
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<String, String> {
        // Try LibreTranslate-compatible API first (self-hosted or public instances)
        let api_url = std::env::var("POLYMDE_TRANSLATE_API_URL")
            .ok()
            .unwrap_or_else(|| "https://libretranslate.com/translate".into());

        let resp = reqwest::Client::new()
            .post(&api_url)
            .json(&serde_json::json!({
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text",
            }))
            .send()
            .await
            .map_err(|e| format!("request failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("API returned {}", resp.status()));
        }

        let body: serde_json::Value = resp.json().await.map_err(|e| format!("parse error: {}", e))?;
        if let Some(translated) = body.get("translatedText").and_then(|t| t.as_str()) {
            return Ok(format!(
                "Translation ({} -> {}):\n{}",
                source_lang, target_lang, translated
            ));
        }

        Err("Unexpected API response format".into())
    }
}

struct SummarizeTool;

#[async_trait::async_trait]
impl Tool for SummarizeTool {
    fn name(&self) -> &str {
        "summarize"
    }

    fn description(&self) -> &str {
        "Summarize a long text, returning key points."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
                "max_sentences": {"type": "integer", "description": "Max sentences in summary, default 3"}
            },
            "required": ["text"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let text = match arg_str_required(args, "text") {
            Ok(t) => t,
            Err(e) => return err_result("summarize", e),
        };
        let max_sentences: usize = arg_u64(args, "max_sentences")
            .map(|v| v as usize)
            .unwrap_or(3);

        // Simple heuristic summary: extract first N sentences
        let sentences: Vec<&str> = text
            .split(|c: char| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim())
            .take(max_sentences)
            .collect();

        if sentences.is_empty() {
            return err_result("summarize", "No sentences found in text".into());
        }

        let summary = sentences.join(". ") + ".";

        tracing::info!(
            input_chars = text.len(),
            summary_chars = summary.len(),
            "summarization complete"
        );

        ok_result("summarize", summary)
    }
}

// ---------------------------------------------------------------------------
// System tools
// ---------------------------------------------------------------------------

struct DateTool;

#[async_trait::async_trait]
impl Tool for DateTool {
    fn name(&self) -> &str {
        "date"
    }

    fn description(&self) -> &str {
        "Get the current date and time, optionally in a specific timezone or format."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "format": {"type": "string", "description": "Date format string, default RFC3339"},
                "timezone": {"type": "string", "description": "Timezone, e.g. 'America/New_York'"}
            },
            "required": []
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let now = chrono::Utc::now();
        let fmt: String = arg_str(args, "format").unwrap_or_else(|| "%Y-%m-%dT%H:%M:%S%.3fZ".into());

        let formatted = now.format(&fmt).to_string();
        ok_result("date", formatted)
    }
}

struct UptimeTool;

#[async_trait::async_trait]
impl Tool for UptimeTool {
    fn name(&self) -> &str {
        "uptime"
    }

    fn description(&self) -> &str {
        "Get system uptime and load averages."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, ctx: &ToolContext, _args: &serde_json::Value) -> ToolResult {
        let result = BashTool.execute(ctx, &serde_json::json!({"command": "uptime"})).await;
        ToolResult {
            name: "uptime".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct DiskUsageTool;

#[async_trait::async_trait]
impl Tool for DiskUsageTool {
    fn name(&self) -> &str {
        "disk_usage"
    }

    fn description(&self) -> &str {
        "Show disk space usage for a path or the filesystem."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to check, default /"},
                "human_readable": {"type": "boolean", "description": "Use human-readable sizes, default true"}
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let path = arg_str(args, "path").unwrap_or_else(|| "/".into());
        let result = BashTool.execute(
            ctx,
            &serde_json::json!({"command": format!("du -sh {}", path)}),
        )
        .await;
        ToolResult {
            name: "disk_usage".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct ProcessListTool;

#[async_trait::async_trait]
impl Tool for ProcessListTool {
    fn name(&self) -> &str {
        "process_list"
    }

    fn description(&self) -> &str {
        "List running processes, optionally filtered."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "filter": {"type": "string", "description": "Filter processes by name"},
                "top": {"type": "integer", "description": "Show top N by CPU, default 20"}
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let filter = arg_str(args, "filter");
        let top: i64 = arg_i64(args, "top").unwrap_or(20);

        let mut cmd = format!("ps aux --sort=-%cpu | head -{}", top + 1);
        if let Some(f) = filter {
            cmd = format!("{} | grep -i {}", cmd, f);
        }

        let result = BashTool.execute(ctx, &serde_json::json!({"command": cmd})).await;
        ToolResult {
            name: "process_list".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct NetworkInfoTool;

#[async_trait::async_trait]
impl Tool for NetworkInfoTool {
    fn name(&self) -> &str {
        "network_info"
    }

    fn description(&self) -> &str {
        "Show network interface information and active connections."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "show_connections": {"type": "boolean", "description": "Also show active connections"}
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let show_conns = arg_bool(args, "show_connections");

        let mut commands = vec!["ip addr show".to_string()];
        if show_conns {
            commands.push("ss -tuln".to_string());
        }

        let result = BashTool.execute(
            ctx,
            &serde_json::json!({"command": commands.join(" && echo --- && ")}),
        )
        .await;
        ToolResult {
            name: "network_info".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

// ---------------------------------------------------------------------------
// Developer tools
// ---------------------------------------------------------------------------

struct GitStatusTool;

#[async_trait::async_trait]
impl Tool for GitStatusTool {
    fn name(&self) -> &str {
        "git_status"
    }

    fn description(&self) -> &str {
        "Show git repository status."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo path, default working directory"}
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let path = arg_str(args, "path");
        let result = BashTool.execute(
            ctx,
            &serde_json::json!({
                "command": if let Some(ref p) = path {
                    format!("git -C {} status --short", p)
                } else {
                    "git status --short".into()
                }
            }),
        )
        .await;
        ToolResult {
            name: "git_status".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct GitLogTool;

#[async_trait::async_trait]
impl Tool for GitLogTool {
    fn name(&self) -> &str {
        "git_log"
    }

    fn description(&self) -> &str {
        "Show git commit log."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of commits, default 10"},
                "path": {"type": "string", "description": "Repo path"}
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let count: i64 = arg_i64(args, "count").unwrap_or(10);
        let path = arg_str(args, "path");
        let result = BashTool.execute(
            ctx,
            &serde_json::json!({
                "command": if let Some(ref p) = path {
                    format!("git -C {} log --oneline -{}", p, count)
                } else {
                    format!("git log --oneline -{}", count)
                }
            }),
        )
        .await;
        ToolResult {
            name: "git_log".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct GitDiffTool;

#[async_trait::async_trait]
impl Tool for GitDiffTool {
    fn name(&self) -> &str {
        "git_diff"
    }

    fn description(&self) -> &str {
        "Show git diff of unstaged changes."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo path"},
                "staged": {"type": "boolean", "description": "Show staged changes instead"}
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let staged = arg_bool(args, "staged");
        let path = arg_str(args, "path");

        let subcmd = if staged { "diff --cached" } else { "diff" };

        let result = BashTool.execute(
            ctx,
            &serde_json::json!({
                "command": if let Some(ref p) = path {
                    format!("git -C {} {}", p, subcmd)
                } else {
                    format!("git {}", subcmd)
                }
            }),
        )
        .await;
        ToolResult {
            name: "git_diff".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct CompilerCheckTool;

#[async_trait::async_trait]
impl Tool for CompilerCheckTool {
    fn name(&self) -> &str {
        "compiler_check"
    }

    fn description(&self) -> &str {
        "Check if a project compiles. Supports Rust (cargo check), Python (pylint), and more."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "language": {"type": "string", "description": "Language: 'rust', 'python', 'typescript', default 'rust'"},
                "path": {"type": "string", "description": "Project path"}
            },
            "required": []
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let language: String = arg_str(args, "language").unwrap_or_else(|| "rust".into());
        let path = arg_str(args, "path");

        let command = match language.as_str() {
            "rust" => "cargo check",
            "python" => "python -m py_compile .",
            "typescript" => "npx tsc --noEmit",
            _ => return err_result("compiler_check", format!("Unsupported language: {}", language)),
        };

        let full_cmd = if let Some(ref p) = path {
            format!("cd {} && {}", p, command)
        } else {
            command.into()
        };

        let result = BashTool.execute(
            ctx,
            &serde_json::json!({"command": full_cmd, "timeout": 120000}),
        )
        .await;
        ToolResult {
            name: "compiler_check".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

struct PackageSearchTool;

#[async_trait::async_trait]
impl Tool for PackageSearchTool {
    fn name(&self) -> &str {
        "package_search"
    }

    fn description(&self) -> &str {
        "Search for packages by registry. Supports crates.io, npm, pypi."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term"},
                "registry": {"type": "string", "description": "Registry: 'crates', 'npm', 'pypi', default 'crates'"}
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let query = match arg_str_required(args, "query") {
            Ok(q) => q,
            Err(e) => return err_result("package_search", e),
        };
        let registry: String = arg_str(args, "registry").unwrap_or_else(|| "crates".into());

        let url = match registry.as_str() {
            "crates" => format!("https://crates.io/api/v1/search?q={}", urlencoding(&query)),
            "npm" => format!(
                "https://registry.npmjs.org/-/v1/search?text={}",
                urlencoding(&query)
            ),
            "pypi" => format!(
                "https://pypi.org/search/?q={}",
                urlencoding(&query)
            ),
            _ => return err_result("package_search", format!("Unknown registry: {}", registry)),
        };

        match reqwest::Client::new()
            .get(&url)
            .header("User-Agent", "Polymede/1.0")
            .send()
            .await
        {
            Ok(resp) => {
                let body = resp.text().await.unwrap_or_default();
                let truncated = if body.len() > 5000 {
                    format!("{}...\n(truncated)", &body[..5000])
                } else {
                    body
                };
                ok_result("package_search", truncated)
            }
            Err(e) => err_result("package_search", format!("Search failed: {e}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Interaction tools
// ---------------------------------------------------------------------------

struct QuestionTool;

#[async_trait::async_trait]
impl Tool for QuestionTool {
    fn name(&self) -> &str {
        "question"
    }

    fn description(&self) -> &str {
        "Ask the user a clarifying question and wait for their response."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to ask the user"},
                "options": {"type": "array", "items": {"type": "string"}, "description": "Optional list of choices"}
            },
            "required": ["question"]
        })
    }

    async fn execute(&self, _ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let question = match arg_str_required(args, "question") {
            Ok(q) => q,
            Err(e) => return err_result("question", e),
        };

        let options = args
            .get("options")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            });

        let prompt = if let Some(ref opts) = options {
            format!(
                "{}\nOptions: [{}]",
                question,
                opts.join(", ")
            )
        } else {
            question
        };

        tracing::info!(question = %prompt, "asking user question");

        ok_result(
            "question",
            format!(
                "Question sent to user: \"{}\". The response will be provided in the next turn.",
                prompt
            ),
        )
    }
}

struct NotifyTool;

#[async_trait::async_trait]
impl Tool for NotifyTool {
    fn name(&self) -> &str {
        "notify"
    }

    fn description(&self) -> &str {
        "Send a desktop notification to the user."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Notification title"},
                "message": {"type": "string", "description": "Notification body text"}
            },
            "required": ["title", "message"]
        })
    }

    async fn execute(&self, ctx: &ToolContext, args: &serde_json::Value) -> ToolResult {
        let title = match arg_str_required(args, "title") {
            Ok(t) => t,
            Err(e) => return err_result("notify", e),
        };
        let message = match arg_str_required(args, "message") {
            Ok(m) => m,
            Err(e) => return err_result("notify", e),
        };

        // Try notify-send, fallback to osascript on macOS
        let cmd = format!(
            "notify-send '{}' '{}' 2>/dev/null || echo 'Notification: {} - {}'",
            title, message, title, message
        );

        let result = BashTool.execute(ctx, &serde_json::json!({"command": cmd})).await;
        ToolResult {
            name: "notify".into(),
            output: result.output,
            ok: result.ok,
        }
    }
}

// ---------------------------------------------------------------------------
// Backward-compatible Tools wrapper
// ---------------------------------------------------------------------------

pub struct Tools {
    registry: ToolRegistry,
    context: ToolContext,
}

impl Tools {
    pub fn new(working_dir: PathBuf, allowed_commands: Option<Vec<String>>) -> Self {
        Self {
            registry: ToolRegistry::new(),
            context: ToolContext::new(working_dir, allowed_commands),
        }
    }

    pub fn definitions() -> Vec<serde_json::Value> {
        ToolRegistry::new().definitions()
    }

    pub async fn execute(&self, call: &ToolCall) -> ToolResult {
        self.registry.execute(&self.context, call).await
    }
}
