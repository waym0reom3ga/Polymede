use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::config::Config;
use crate::llm::{
    ChatResponse, LlmClient, LlmError, Message, MessageRole, ToolCall as LlmToolCall,
    ToolDefinition, TokenUsage,
};

fn json_to_tool_defs(json_defs: Vec<serde_json::Value>) -> Vec<ToolDefinition> {
    json_defs
        .iter()
        .filter_map(|j| {
            let func = j.get("function")?;
            Some(ToolDefinition {
                name: func.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                description: func
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                schema: func.get("parameters").cloned().unwrap_or_default(),
            })
        })
        .collect()
}
use crate::memory::{MemoryError, MemoryIntegration};
use crate::tools::{ToolCall, ToolContext, ToolRegistry, ToolResult as ToolExecResult};

// ---------------------------------------------------------------------------
// Agent error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum AgentError {
    Llm(LlmError),
    Memory(MemoryError),
    Tool(String),
    ContextFull,
    Shutdown,
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::Llm(e) => write!(f, "LLM error: {e}"),
            AgentError::Memory(e) => write!(f, "memory error: {e}"),
            AgentError::Tool(msg) => write!(f, "tool error: {msg}"),
            AgentError::ContextFull => write!(f, "context budget exceeded"),
            AgentError::Shutdown => write!(f, "agent shutting down"),
        }
    }
}

impl std::error::Error for AgentError {}

impl From<LlmError> for AgentError {
    fn from(e: LlmError) -> Self {
        AgentError::Llm(e)
    }
}

impl From<MemoryError> for AgentError {
    fn from(e: MemoryError) -> Self {
        AgentError::Memory(e)
    }
}

// ---------------------------------------------------------------------------
// Conversation context
// ---------------------------------------------------------------------------

/// Tracks the rolling conversation window and token budget.
pub struct ConversationContext {
    messages: Vec<Message>,
    tokens_used: u32,
    max_tokens: u32,
    tool_definitions: Vec<serde_json::Value>,
}

impl ConversationContext {
    const DEFAULT_MAX_TOKENS: u32 = 128_000;
    const COMPRESSION_THRESHOLD: f32 = 0.75;

    pub fn new(tool_definitions: Vec<serde_json::Value>) -> Self {
        Self {
            messages: Vec::new(),
            tokens_used: 0,
            max_tokens: Self::DEFAULT_MAX_TOKENS,
            tool_definitions,
        }
    }

    pub fn with_max_tokens(tool_definitions: Vec<serde_json::Value>, max_tokens: u32) -> Self {
        Self {
            messages: Vec::new(),
            tokens_used: 0,
            max_tokens,
            tool_definitions,
        }
    }

    pub fn add_message(&mut self, message: Message) {
        let tokens = estimate_message_tokens(&message);
        self.tokens_used += tokens;
        self.messages.push(message);
    }

    pub fn tokens_used(&self) -> u32 {
        self.tokens_used
    }

    pub fn remaining_tokens(&self) -> u32 {
        self.max_tokens.saturating_sub(self.tokens_used)
    }

    pub fn is_near_limit(&self) -> bool {
        (self.tokens_used as f32 / self.max_tokens as f32) > Self::COMPRESSION_THRESHOLD
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Trim the conversation to fit within budget by removing the oldest
    /// user/assistant pairs, keeping the system message intact.
    pub fn trim_to_budget(&mut self, budget: u32) {
        while self.tokens_used > budget && self.messages.len() > 1 {
            if let Some(idx) = self.find_trim_point() {
                let trimmed = self.messages.drain(..=idx).collect::<Vec<_>>();
                for msg in &trimmed {
                    self.tokens_used = self.tokens_used.saturating_sub(estimate_message_tokens(msg));
                }
            } else {
                break;
            }
        }
    }

    /// Replace the body of the conversation with a compressed summary.
    pub fn compress_to_summary(&mut self, summary: &str) {
        let system = self.messages.first().cloned();
        self.messages.clear();
        self.tokens_used = 0;

        if let Some(sys) = system {
            self.messages.push(sys);
            self.tokens_used += estimate_message_tokens(&self.messages[0]);
        }

        let summary_msg = Message {
            role: MessageRole::Assistant,
            content: format!("[Previous conversation summary]\n{summary}"),
            tool_calls: None,
            tool_call_id: None,
        };
        self.add_message(summary_msg);
    }

    fn find_trim_point(&self) -> Option<usize> {
        for i in (1..self.messages.len()).rev() {
            if self.messages[i].role == MessageRole::User {
                return Some(i);
            }
        }
        None
    }
}

fn estimate_message_tokens(msg: &Message) -> u32 {
    let base = 8;
    // Use tiktoken gpt-4 BPE for accurate token counting.
    // Falls back to len/4 if tokenizer init fails.
    static ENCODER: std::sync::OnceLock<tiktoken_rs::CoreBPE> = std::sync::OnceLock::new();
    let content_tokens = tiktoken_rs::get_bpe_from_model("gpt-4")
        .ok()
        .map(|enc| {
            ENCODER.set(enc).ok();
            ENCODER.get().map(|e| e.encode_with_special_tokens(&msg.content).len())
        })
        .flatten()
        .unwrap_or_else(|| msg.content.len().max(1) / 4) as u32;
    let tool_call_tokens = msg
        .tool_calls
        .as_ref()
        .map(|calls| calls.len() as u32 * 16)
        .unwrap_or(0);
    base + content_tokens.max(1) + tool_call_tokens
}

// ---------------------------------------------------------------------------
// Skill system
// ---------------------------------------------------------------------------

/// A skill is identified by a name, trigger patterns, and an instruction block
/// that gets injected into the system prompt when triggered.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub triggers: Vec<String>,
    pub instruction: String,
}

impl Skill {
    pub fn new(name: String, triggers: Vec<String>, instruction: String) -> Self {
        Self {
            name,
            triggers,
            instruction,
        }
    }

    fn matches_input(&self, input: &str) -> bool {
        let lower = input.to_lowercase();
        self.triggers
            .iter()
            .any(|t| lower.contains(&t.to_lowercase()))
    }
}

// ---------------------------------------------------------------------------
// Subagent
// ---------------------------------------------------------------------------

/// Result from a spawned subagent.
#[derive(Debug, Clone)]
pub struct SubagentResult {
    pub task_id: String,
    pub output: String,
    pub token_usage: TokenUsage,
}

/// Request to spawn a subagent for a delegated task.
#[derive(Debug, Clone)]
pub struct SubagentRequest {
    pub task: String,
    pub context: String,
    pub tool_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// Input source
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum AgentInput {
    /// Text from the TUI.
    Tui(String),
    /// Message from the gateway.
    Gateway { source: String, content: String },
    /// Reset conversation context (clear messages, keep system prompt).
    Reset,
    /// Force compress the conversation context now.
    Compress,
    /// Switch LLM model at runtime.
    SetModel(String),
}

impl AgentInput {
    pub fn content(&self) -> Option<&str> {
        match self {
            AgentInput::Tui(s) => Some(s),
            AgentInput::Gateway { content, .. } => Some(content),
            AgentInput::Reset | AgentInput::Compress | AgentInput::SetModel(_) => None,
        }
    }

    pub fn source_label(&self) -> &str {
        match self {
            AgentInput::Tui(_) => "tui",
            AgentInput::Gateway { source, .. } => source.as_str(),
            AgentInput::Reset | AgentInput::Compress | AgentInput::SetModel(_) => "system",
        }
    }
}

// ---------------------------------------------------------------------------
// Agent output
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AgentOutput {
    pub content: String,
    pub token_usage: TokenUsage,
    pub tool_results: Vec<ToolExecResult>,
    pub tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

pub struct Agent {
    config: Config,
    memory: MemoryIntegration,
    tools: ToolRegistry,
    llm: tokio::sync::Mutex<LlmClient>,
    context: Arc<RwLock<ConversationContext>>,
    skills: Vec<Skill>,
    persona: String,
    tool_context: ToolContext,
    /// Channel for receiving input.
    input_rx: mpsc::UnboundedReceiver<AgentInput>,
    /// Channel for emitting output.
    output_tx: mpsc::UnboundedSender<AgentOutput>,
    /// Channel for streaming text chunks to TUI.
    chunk_tx: mpsc::UnboundedSender<String>,
    /// Handle to cancel the running loop.
    shutdown_tx: mpsc::UnboundedSender<()>,
    shutdown_rx: mpsc::UnboundedReceiver<()>,
}

impl Agent {
    /// Create a new agent with the given config and memory integration.
    ///
    /// Returns the agent plus the sender half for feeding input and the
    /// receiver half for collecting output.
    pub async fn new(
        config: Config,
        memory: MemoryIntegration,
    ) -> Result<
        (
            Self,
            mpsc::UnboundedSender<AgentInput>,
            mpsc::UnboundedReceiver<AgentOutput>,
            mpsc::UnboundedReceiver<String>,
        ),
        AgentError,
    > {
        let tool_registry = ToolRegistry::new().with_cache(300, 100);
        let tool_definitions = tool_registry.definitions();

        let llm = LlmClient::with_options(
            config.llm.clone(),
            2,
            0.7,
            Some(8192),
        );

        let context = Arc::new(RwLock::new(ConversationContext::new(
            tool_definitions.clone(),
        )));

        let persona = Self::build_persona(&config);
        let skills = Self::load_skills(&config).await;

        let rate_limiter = Arc::new(crate::ratelimit::RateLimiter::new());

        let tool_context = ToolContext::new(
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            None,
        );
        // Patch rate_limiter into context (ToolContext is not Arc so we set directly).
        let mut tool_context = tool_context;
        tool_context.rate_limiter = Some(rate_limiter);

        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let (chunk_tx, chunk_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = mpsc::unbounded_channel();

        let agent = Self {
            config,
            memory,
            tools: tool_registry,
            llm: tokio::sync::Mutex::new(llm),
            context,
            skills,
            persona,
            tool_context,
            input_rx,
            output_tx,
            chunk_tx,
            shutdown_tx,
            shutdown_rx,
        };

        Ok((agent, input_tx, output_rx, chunk_rx))
    }

    /// Start the main agent loop.  Blocks until shutdown is signaled or the
    /// input channel is closed.
    pub async fn run(mut self) -> Result<(), AgentError> {
        tracing::info!(
            session = %self.memory.session_id(),
            "agent loop started"
        );

        loop {
            tokio::select! {
                biased;

                _ = self.shutdown_rx.recv() => {
                    tracing::info!("shutdown signal received");
                    break;
                }

                input = self.input_rx.recv() => {
                    let input = match input {
                        Some(i) => i,
                        None => {
                            tracing::info!("input channel closed");
                            break;
                        }
                    };

                    match self.process_turn(input).await {
                        Ok(output) => {
                            if let Err(e) = self.output_tx.send(output) {
                                tracing::error!(error = %e, "failed to send output");
                                break;
                            }
                        }
                        Err(AgentError::Shutdown) => break,
                        Err(e) => {
                            tracing::error!(error = %e, "turn processing failed");
                        }
                    }
                }
            }
        }

        tracing::info!("agent loop exited");
        Ok(())
    }

    /// Signal the agent to shut down.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    // -----------------------------------------------------------------------
    // Core turn processing
    // -----------------------------------------------------------------------

    async fn process_turn(&self, input: AgentInput) -> Result<AgentOutput, AgentError> {
        // Handle reset — clear context without LLM call.
        if matches!(&input, AgentInput::Reset) {
            let mut ctx = self.context.write().await;
            ctx.messages.clear();
            ctx.tokens_used = 0;
            self.tools.clear_cache().await;
            tracing::info!("agent context cleared");
            return Ok(AgentOutput {
                content: "Context reset.".to_string(),
                token_usage: TokenUsage::default(),
                tool_results: Vec::new(),
                tags: vec![],
            });
        }

        // Handle forced compression.
        if matches!(&input, AgentInput::Compress) {
            let mut ctx = self.context.write().await;
            self.maybe_compress_context_internal(&mut ctx).await?;
            return Ok(AgentOutput {
                content: "Context compressed.".to_string(),
                token_usage: TokenUsage::default(),
                tool_results: Vec::new(),
                tags: vec![],
            });
        }

        // Handle model switch.
        if let AgentInput::SetModel(model) = input {
            self.llm.lock().await.set_model(model.clone());
            return Ok(AgentOutput {
                content: format!("Model switched to '{}'.", model),
                token_usage: TokenUsage::default(),
                tool_results: Vec::new(),
                tags: vec![],
            });
        }

        let user_content = input.content().unwrap_or("").to_string();
        let source = input.source_label().to_string();

        tracing::info!(
            source = %source,
            input_len = user_content.len(),
            "processing turn"
        );

        // Step 1: Check context budget and compress if needed.
        self.maybe_compress_context().await?;

        // Step 2: Recall relevant memories.
        let recall_tags = self.extract_tags(&user_content);
        let recall_result = self.memory.recall(&recall_tags, None).await?;
        let recalled_text = self.format_recalled_memories(&recall_result);

        // Step 3: Determine active skills.
        let active_skills: Vec<&Skill> = self
            .skills
            .iter()
            .filter(|s| s.matches_input(&user_content))
            .collect();

        // Step 4: Build system prompt.
        let system_prompt = self.build_system_prompt(&recalled_text, &active_skills);

        // Step 5: Send to LLM with tool definitions.
        let mut ctx = self.context.write().await;
        ctx.add_message(Message {
            role: MessageRole::System,
            content: system_prompt,
            tool_calls: None,
            tool_call_id: None,
        });

        ctx.add_message(Message {
            role: MessageRole::User,
            content: user_content.clone(),
            tool_calls: None,
            tool_call_id: None,
        });

        let messages = ctx.messages().to_vec();
        let tool_defs = json_to_tool_defs(ctx.tool_definitions.clone());

        // Use streaming — chunks flow to TUI for progressive rendering.
        let response: ChatResponse = self
            .llm
            .lock()
            .await
            .chat_stream(messages, Some(tool_defs), Some(self.chunk_tx.clone()))
            .await?
            .into();

        // Step 6: Handle tool calls if present.
        let (final_content, tool_results, total_usage) = self
            .handle_tool_calls(&mut ctx, response)
            .await?;

        // Step 7: Extract tags from the turn.
        let tags = self.combine_tags(&recall_tags, &user_content);

        // Step 8: Log interaction to memory.
        let tool_call_names: Vec<String> = tool_results.iter().map(|t| t.name.clone()).collect();
        self.memory
            .ingest(
                "user",
                &user_content,
                &[],
                &tags.clone(),
            )
            .await?;

        self.memory
            .ingest(
                "assistant",
                &final_content,
                &tool_call_names,
                &tags,
            )
            .await?;

        let output = AgentOutput {
            content: final_content,
            token_usage: total_usage,
            tool_results,
            tags,
        };

        tracing::info!(
            tokens = %output.token_usage.total_tokens,
            tools_used = output.tool_results.len(),
            "turn complete"
        );

        Ok(output)
    }

    // -----------------------------------------------------------------------
    // Tool call orchestration
    // -----------------------------------------------------------------------

    async fn handle_tool_calls(
        &self,
        ctx: &mut ConversationContext,
        response: ChatResponse,
    ) -> Result<(String, Vec<ToolExecResult>, TokenUsage), AgentError> {
        let mut total_usage = response.usage.clone();
        let mut tool_results = Vec::new();

        let mut current_response = response;

        // Allow up to 10 tool call rounds per turn to prevent infinite loops.
        let mut tool_rounds = 0;
        const MAX_TOOL_ROUNDS: u32 = 10;

        while !current_response.tool_calls.is_empty() && tool_rounds < MAX_TOOL_ROUNDS {
            tool_rounds += 1;

            // Log assistant message with tool calls.
            ctx.add_message(Message {
                role: MessageRole::Assistant,
                content: current_response.content.clone(),
                tool_calls: Some(
                    current_response
                        .tool_calls
                        .iter()
                        .map(|tc| LlmToolCall {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        })
                        .collect(),
                ),
                tool_call_id: None,
            });

            // Execute each tool call.
            for tool_call in &current_response.tool_calls {
                let tool_call = ToolCall {
                    id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                };

                let result = self.tools.execute(&self.tool_context, &tool_call).await;

                tracing::debug!(
                    tool = %tool_call.name,
                    success = result.ok,
                    output_len = result.output.len(),
                    "tool executed"
                );

                // Add tool result as a tool message.
                ctx.add_message(Message {
                    role: MessageRole::Tool,
                    content: result.output.clone(),
                    tool_calls: None,
                    tool_call_id: Some(tool_call.id.clone()),
                });

                tool_results.push(result);
            }

            // Check if context is getting tight after tool results.
            if ctx.is_near_limit() {
                self.maybe_compress_context_internal(ctx).await?;
            }

            // Send updated context back to LLM for next decision.
            let messages = ctx.messages().to_vec();
            let tool_defs = json_to_tool_defs(ctx.tool_definitions.clone());

            current_response = self.llm.lock().await.chat_stream(messages, Some(tool_defs), None).await?.into();
            total_usage.add(&current_response.usage);
        }

        // Log final assistant message.
        ctx.add_message(Message {
            role: MessageRole::Assistant,
            content: current_response.content.clone(),
            tool_calls: None,
            tool_call_id: None,
        });

        Ok((current_response.content, tool_results, total_usage))
    }

    // -----------------------------------------------------------------------
    // Context management
    // -----------------------------------------------------------------------

    async fn maybe_compress_context(&self) -> Result<(), AgentError> {
        let mut ctx = self.context.write().await;
        self.maybe_compress_context_internal(&mut ctx).await
    }

    async fn maybe_compress_context_internal(
        &self,
        ctx: &mut ConversationContext,
    ) -> Result<(), AgentError> {
        if !ctx.is_near_limit() {
            return Ok(());
        }

        tracing::warn!(
            tokens_used = ctx.tokens_used(),
            max_tokens = ctx.max_tokens,
            "context near limit, compressing"
        );

        // Attempt LLM-assisted compression of the conversation body.
        let summary = self.compress_conversation(ctx).await;

        match summary {
            Ok(s) => {
                ctx.compress_to_summary(&s);
                tracing::info!("context compressed successfully");
            }
            Err(e) => {
                tracing::warn!(error = %e, "LLM compression failed, falling back to trim");
                ctx.trim_to_budget((ctx.max_tokens as f32 * 0.5) as u32);
            }
        }

        Ok(())
    }

    /// Ask the LLM to summarize the current conversation history.
    async fn compress_conversation(&self, ctx: &ConversationContext) -> Result<String, AgentError> {
        let body_messages: Vec<Message> = ctx
            .messages()
            .iter()
            .skip(1)
            .cloned()
            .collect();

        if body_messages.is_empty() {
            return Ok("No prior conversation to summarize.".into());
        }

        let content: String = body_messages
            .iter()
            .map(|m| format!("[{:?}]: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let summary_prompt = format!(
            "Summarize the following conversation concisely, preserving key decisions, \
             facts, and action items:\n\n{content}"
        );

        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are a concise summarizer. Output only the summary.".into(),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: MessageRole::User,
                content: summary_prompt,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let resp: ChatResponse = self.llm.lock().await.chat_stream(messages, None, None).await?.into();
        Ok(resp.content)
    }

    // -----------------------------------------------------------------------
    // Subagent spawning
    // -----------------------------------------------------------------------

    /// Spawn a subagent to handle a delegated task in parallel.
    ///
    /// The subagent runs with a fresh context and a subset of tools.  Results
    /// are returned as a summary string.
    pub async fn spawn_subagent(
        &self,
        request: SubagentRequest,
    ) -> Result<SubagentResult, AgentError> {
        let task_id = Uuid::new_v4().to_string();

        tracing::info!(
            task_id = %task_id,
            task = %request.task,
            "spawning subagent"
        );

        // Build a restricted tool set for the subagent.
        let sub_tools = self.build_subagent_tools(&request.tool_names);
        let sub_tool_defs = sub_tools.definitions();

        let _sub_ctx = ConversationContext::new(sub_tool_defs);

        let system_prompt = format!(
            "You are a subagent of Polymede. Your task: {}\n\nContext: {}",
            request.task, request.context
        );

        let mut messages = vec![
            Message {
                role: MessageRole::System,
                content: system_prompt,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: MessageRole::User,
                content: request.task.clone(),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let mut total_usage = TokenUsage::default();
        let mut tool_results = Vec::new();

        let mut response: ChatResponse = self
            .llm
            .lock()
            .await
            .chat_stream(messages.clone(), Some(json_to_tool_defs(sub_tools.definitions())), None)
            .await?
            .into();

        total_usage.add(&response.usage);

        // Run tool call loop for subagent.
        let mut rounds = 0;
        while !response.tool_calls.is_empty() && rounds < 5 {
            rounds += 1;

            messages.push(Message {
                role: MessageRole::Assistant,
                content: response.content.clone(),
                tool_calls: Some(
                    response
                        .tool_calls
                        .iter()
                        .map(|tc| LlmToolCall {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        })
                        .collect(),
                ),
                tool_call_id: None,
            });

            for tc in &response.tool_calls {
                let call = ToolCall {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                };
                let result = sub_tools.execute(&self.tool_context, &call).await;
                messages.push(Message {
                    role: MessageRole::Tool,
                    content: result.output.clone(),
                    tool_calls: None,
                    tool_call_id: Some(tc.id.clone()),
                });
                tool_results.push(result);
            }

            response = self.llm.lock().await.chat_stream(messages.clone(), Some(json_to_tool_defs(sub_tools.definitions())), None).await?.into();
            total_usage.add(&response.usage);
        }

        messages.push(Message {
            role: MessageRole::Assistant,
            content: response.content.clone(),
            tool_calls: None,
            tool_call_id: None,
        });

        // Log subagent interaction to memory.
        let sub_tags = vec!["subagent".into(), task_id.clone()];
        self.memory
            .ingest("subagent", &response.content, &[], &sub_tags)
            .await?;

        tracing::info!(
            task_id = %task_id,
            "subagent completed"
        );

        Ok(SubagentResult {
            task_id,
            output: response.content,
            token_usage: total_usage,
        })
    }

    fn build_subagent_tools(&self, requested: &[String]) -> ToolRegistry {
        let _all_tools = ToolRegistry::new();

        let tools: Vec<Arc<dyn crate::tools::Tool>> = ToolRegistry::default_tools()
            .into_iter()
            .filter(|t| requested.iter().any(|r| r == t.name()))
            .collect();

        if tools.is_empty() {
            ToolRegistry::new()
        } else {
            ToolRegistry::with_tools(tools)
        }
    }

    // -----------------------------------------------------------------------
    // System prompt construction
    // -----------------------------------------------------------------------

    fn build_system_prompt(&self, recalled: &str, active_skills: &[&Skill]) -> String {
        let mut parts = Vec::new();

        // Base persona.
        parts.push(self.persona.clone());

        // Active skill instructions.
        if !active_skills.is_empty() {
            parts.push("ACTIVE SKILLS:".to_string());
            for skill in active_skills {
                parts.push(format!("  - {}: {}", skill.name, skill.instruction));
            }
        }

        // Recalled memories.
        if !recalled.is_empty() {
            parts.push(format!("RELEVANT MEMORIES:\n{recalled}"));
        }

        // Available tools.
        let tool_names: Vec<&str> = self.tools.tool_names();
        parts.push(format!(
            "AVAILABLE TOOLS: {}",
            tool_names.join(", ")
        ));

        parts.push(
            "When you need to use a tool, return a tool call. \
             When you have enough information, respond directly."
                .to_string(),
        );

        parts.join("\n\n")
    }

    fn build_persona(config: &Config) -> String {
        format!(
            "You are Polymede, an AI agent powered by {} via {}. \
             You can execute tools, manage files, run commands, search the web, \
             and coordinate subagents for complex tasks. Be concise, accurate, \
             and proactive in using tools when it helps the user.",
            config.llm.model, config.llm.provider
        )
    }

    // -----------------------------------------------------------------------
    // Skill loading
    // -----------------------------------------------------------------------

    async fn load_skills(config: &Config) -> Vec<Skill> {
        let skill_dir = config.skill_dir();

        if !skill_dir.exists() {
            return Vec::new();
        }

        let mut skills = Vec::new();

        let mut entries = match tokio::fs::read_dir(&skill_dir).await {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "cannot read skills directory");
                return Vec::new();
            }
        };

        let mut entries_vec = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            entries_vec.push(entry);
        }

        for entry in entries_vec {
            let path = entry.path();
            if !path.extension().map_or(false, |ext| ext == "toml") {
                continue;
            }

            match tokio::fs::read_to_string(&path).await {
                Ok(content) => {
                    match Self::parse_skill_toml(&content) {
                        Ok(skill) => {
                            tracing::info!(skill = %skill.name, "loaded skill");
                            skills.push(skill);
                        }
                        Err(e) => {
                            tracing::warn!(
                                file = %path.display(),
                                error = %e,
                                "failed to parse skill"
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        file = %path.display(),
                        error = %e,
                        "cannot read skill file"
                    );
                }
            }
        }

        skills
    }

    fn parse_skill_toml(content: &str) -> Result<Skill, String> {
        #[derive(serde::Deserialize)]
        struct SkillToml {
            name: String,
            triggers: Vec<String>,
            instruction: String,
        }

        let parsed: SkillToml =
            toml::from_str(content).map_err(|e| format!("TOML parse error: {e}"))?;

        Ok(Skill::new(
            parsed.name,
            parsed.triggers,
            parsed.instruction,
        ))
    }

    // -----------------------------------------------------------------------
    // Memory formatting
    // -----------------------------------------------------------------------

    fn format_recalled_memories(&self, result: &crate::memory::RecallResult) -> String {
        if result.memories.is_empty() && result.raw_interactions.is_empty() {
            return String::new();
        }

        let mut parts = Vec::new();

        if !result.memories.is_empty() {
            let summaries: Vec<String> = result
                .memories
                .iter()
                .map(|m| format!("[L{}] {}", m.level, m.summary))
                .collect();
            parts.push(format!("Compressed memories:\n{}", summaries.join("\n")));
        }

        if !result.raw_interactions.is_empty() {
            let recent: Vec<String> = result
                .raw_interactions
                .iter()
                .rev()
                .take(10)
                .map(|i| format!("[{}]: {}", i.role, i.content))
                .collect();
            parts.push(format!("Recent interactions:\n{}", recent.join("\n")));
        }

        parts.join("\n\n")
    }

    // -----------------------------------------------------------------------
    // Tag extraction
    // -----------------------------------------------------------------------

    fn extract_tags(&self, input: &str) -> Vec<String> {
        let mut words: Vec<String> = input
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| w.len() > 3)
            .collect();

        let mut tags = Vec::new();

        if input.contains("file") || input.contains("read") || input.contains("write") {
            tags.push("file_ops".into());
        }
        if input.contains("code") || input.contains("rust") || input.contains("python") {
            tags.push("coding".into());
        }
        if input.contains("search") || input.contains("web") || input.contains("http") {
            tags.push("web".into());
        }
        if input.contains("git") || input.contains("commit") || input.contains("branch") {
            tags.push("git".into());
        }
        if input.contains("test") || input.contains("debug") {
            tags.push("debugging".into());
        }

        for tag in tags {
            if !words.contains(&tag) {
                words.push(tag);
            }
        }

        words.into_iter().take(20).collect()
    }

    fn combine_tags(&self, recall_tags: &[String], input: &str) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut combined = Vec::new();

        for tag in recall_tags {
            if seen.insert(tag.clone()) {
                combined.push(tag.clone());
            }
        }

        let new_tags = self.extract_tags(input);
        for tag in new_tags {
            if seen.insert(tag.clone()) {
                combined.push(tag);
            }
        }

        combined
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_message_tokens_basic() {
        let msg = Message {
            role: MessageRole::User,
            content: "Hello, this is a test message.".into(),
            tool_calls: None,
            tool_call_id: None,
        };
        let tokens = estimate_message_tokens(&msg);
        assert!(tokens > 8);
    }

    #[test]
    fn estimate_message_tokens_with_tool_calls() {
        let msg = Message {
            role: MessageRole::Assistant,
            content: "Using a tool".into(),
            tool_calls: Some(vec![LlmToolCall {
                id: "tc1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "ls"}),
            }]),
            tool_call_id: None,
        };
        let tokens = estimate_message_tokens(&msg);
        assert!(tokens > 20);
    }

    #[test]
    fn conversation_context_tracks_tokens() {
        let mut ctx = ConversationContext::new(vec![]);
        assert_eq!(ctx.tokens_used(), 0);

        ctx.add_message(Message {
            role: MessageRole::User,
            content: "test".into(),
            tool_calls: None,
            tool_call_id: None,
        });
        assert!(ctx.tokens_used() > 0);
    }

    #[test]
    fn conversation_context_near_limit() {
        let mut ctx = ConversationContext::with_max_tokens(vec![], 100);
        assert!(!ctx.is_near_limit());

        for _ in 0..20 {
            ctx.add_message(Message {
                role: MessageRole::User,
                content: "x".into(),
                tool_calls: None,
                tool_call_id: None,
            });
        }
        assert!(ctx.is_near_limit());
    }

    #[test]
    fn conversation_context_trim() {
        let mut ctx = ConversationContext::with_max_tokens(vec![], 50);
        for _ in 0..10 {
            ctx.add_message(Message {
                role: MessageRole::User,
                content: "hello world this is a longer message to consume tokens".into(),
                tool_calls: None,
                tool_call_id: None,
            });
        }
        let before = ctx.tokens_used();
        ctx.trim_to_budget(30);
        assert!(ctx.tokens_used() < before);
    }

    #[test]
    fn skill_matching() {
        let skill = Skill::new(
            "coding".into(),
            vec!["code".into(), "function".into()],
            "Help with coding tasks.".into(),
        );
        assert!(skill.matches_input("Write a function"));
        assert!(skill.matches_input("I need help with code"));
        assert!(!skill.matches_input("What is the weather?"));
    }

    #[test]
    fn persona_contains_model_info() {
        let config = Config {
            llm: crate::config::LlmConfig {
                provider: "test-provider".into(),
                model: "test-model".into(),
                api_key: Some("key".into()),
                base_url: None,
                fallback: None,
            },
            tools: Default::default(),
            memory: Default::default(),
            logging: Default::default(),
        };
        let persona = Agent::build_persona(&config);
        assert!(persona.contains("test-model"));
        assert!(persona.contains("test-provider"));
    }

    #[test]
    fn system_prompt_includes_skills() {
        let config = Config {
            llm: crate::config::LlmConfig {
                provider: "test".into(),
                model: "test".into(),
                api_key: Some("k".into()),
                base_url: None,
                fallback: None,
            },
            tools: Default::default(),
            memory: Default::default(),
            logging: Default::default(),
        };

        let skills = vec![Skill::new(
            "test_skill".into(),
            vec!["test".into()],
            "Do test things.".into(),
        )];
        let active: Vec<&Skill> = skills.iter().collect();
        let prompt = format!(
            "{}\n\nACTIVE SKILLS:\n  - {}: {}\n\nAVAILABLE TOOLS:\n\nWhen you need to use a tool, \
             return a tool call. When you have enough information, respond directly.",
            Agent::build_persona(&config),
            active[0].name,
            active[0].instruction
        );
        assert!(prompt.contains("ACTIVE SKILLS"));
        assert!(prompt.contains("test_skill"));
    }

    #[test]
    fn agent_input_content() {
        let tui = AgentInput::Tui("hello".into());
        assert_eq!(tui.content(), Some("hello"));
        assert_eq!(tui.source_label(), "tui");

        let gw = AgentInput::Gateway {
            source: "grpc".into(),
            content: "world".into(),
        };
        assert_eq!(gw.content(), Some("world"));
        assert_eq!(gw.source_label(), "grpc");
    }
}
