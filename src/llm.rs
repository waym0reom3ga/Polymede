use tokio_stream::StreamExt;
use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use crate::config::LlmConfig;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "parameters")]
    pub schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    #[serde(rename = "arguments")]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn add(&mut self, other: &TokenUsage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
    }
}

#[derive(Debug)]
pub enum LlmError {
    Http(reqwest::StatusCode, String),
    Network(reqwest::Error),
    Deserialize(serde_json::Error),
    NoApiKey,
    FallbackExhausted(Vec<String>),
    InvalidResponse(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::Http(code, msg) => write!(f, "HTTP {code}: {msg}"),
            LlmError::Network(e) => write!(f, "network error: {e}"),
            LlmError::Deserialize(e) => write!(f, "deserialize error: {e}"),
            LlmError::NoApiKey => write!(f, "no API key configured"),
            LlmError::FallbackExhausted(errors) => {
                write!(f, "all providers failed:\n{}", errors.join("\n"))
            }
            LlmError::InvalidResponse(msg) => write!(f, "invalid response: {msg}"),
        }
    }
}

impl std::error::Error for LlmError {}

impl From<reqwest::Error> for LlmError {
    fn from(e: reqwest::Error) -> Self {
        if let Some(status) = e.status() {
            LlmError::Http(status, e.to_string())
        } else {
            LlmError::Network(e)
        }
    }
}

// ---------------------------------------------------------------------------
// LlmClient
// ---------------------------------------------------------------------------

pub struct LlmClient {
    config: LlmConfig,
    http: reqwest::Client,
    max_retries: u32,
    temperature: f32,
    max_tokens: Option<u32>,
}

impl LlmClient {
    const DEFAULT_MAX_RETRIES: u32 = 2;
    const DEFAULT_TEMPERATURE: f32 = 0.7;
    const RETRY_DELAY_SECS: u64 = 2;

    pub fn new(config: LlmConfig) -> Self {
        Self::with_options(config, Self::DEFAULT_MAX_RETRIES, Self::DEFAULT_TEMPERATURE, None)
    }

    pub fn with_options(
        config: LlmConfig,
        max_retries: u32,
        temperature: f32,
        max_tokens: Option<u32>,
    ) -> Self {
        LlmClient {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("build reqwest client"),
            config,
            max_retries,
            temperature,
            max_tokens,
        }
    }

    /// Change the model at runtime.
    pub fn set_model(&mut self, model: String) {
        tracing::info!(model = %model, "model switched");
        self.config.model = model;
    }

    /// Return the current model name.
    pub fn model_name(&self) -> &str {
        &self.config.model
    }

    // -- Public API ----------------------------------------------------------

    /// Send a chat completion request and return the full response.
    pub async fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut errors = Vec::new();
        let mut current = Some(self.config.clone());

        while let Some(cfg) = current {
            match self.chat_with_config(&cfg, &messages, &tools).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    tracing::warn!(provider = %cfg.provider, error = %e, "provider failed");
                    errors.push(format!("[{}] {}", cfg.provider, e));
                    current = cfg.fallback.as_ref().map(Box::as_ref).cloned();
                }
            }
        }

        Err(LlmError::FallbackExhausted(errors))
    }

    /// Stream a chat completion, yielding chunks as they arrive.
    pub async fn chat_stream(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        chunk_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<StreamResult, LlmError> {
        let mut errors = Vec::new();
        let mut current = Some(self.config.clone());

        while let Some(cfg) = current {
            match self.chat_stream_inner(&cfg, &messages, &tools, chunk_tx.as_ref()).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    tracing::warn!(provider = %cfg.provider, error = %e, "stream provider failed");
                    errors.push(format!("[{}] {}", cfg.provider, e));
                    current = cfg.fallback.as_ref().map(Box::as_ref).cloned();
                }
            }
        }

        Err(LlmError::FallbackExhausted(errors))
    }

    // -- Internal ------------------------------------------------------------

    async fn chat_with_config(
        &self,
        config: &LlmConfig,
        messages: &[Message],
        tools: &Option<Vec<ToolDefinition>>,
    ) -> Result<ChatResponse, LlmError> {
        let api_key = self.effective_key(config)?;
        let (base_url, is_anthropic) = self.resolve_endpoint(config)?;
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let delay = Duration::from_secs(Self::RETRY_DELAY_SECS * attempt as u64);
                tracing::info!(attempt, delay_sec = ?delay, "retrying");
                tokio::time::sleep(delay).await;
            }

            let body = if is_anthropic {
                self.build_anthropic_body(config, messages, tools, &api_key)?
            } else {
                self.build_openai_body(config, messages, tools)?
            };

            let url = format!("{base_url}/chat/completions");
            tracing::debug!(url, model = %config.model, "request");

            let resp = self
                .http
                .post(&url)
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", api_key))
                .json(&body)
                .send()
                .await;

            match resp {
                Ok(response) => {
                    if !response.status().is_success() {
                        let status = response.status();
                        let body_text = response.text().await.unwrap_or_default();
                        if status.as_u16() < 500 || attempt == self.max_retries {
                            return Err(LlmError::Http(status, body_text));
                        }
                        last_error = Some(LlmError::Http(status, body_text));
                        continue;
                    }

                    let resp = if is_anthropic {
                        self.parse_anthropic_usage(response).await?
                    } else {
                        self.parse_openai_usage(response).await?
                    };

                    return Ok(resp);
                }
                Err(e) => {
                    let err = LlmError::from(e);
                    if attempt == self.max_retries {
                        return Err(err);
                    }
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| LlmError::Http(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "max retries exceeded".into(),
        )))
    }

    async fn chat_stream_inner(
        &self,
        config: &LlmConfig,
        messages: &[Message],
        tools: &Option<Vec<ToolDefinition>>,
        chunk_tx: Option<&mpsc::UnboundedSender<String>>,
    ) -> Result<StreamResult, LlmError> {
        let api_key = self.effective_key(config)?;
        let (base_url, is_anthropic) = self.resolve_endpoint(config)?;

        let body = if is_anthropic {
            self.build_anthropic_stream_body(config, messages, tools, &api_key)?
        } else {
            self.build_openai_stream_body(config, messages, tools)?
        };

        let url = format!("{base_url}/chat/completions");
        tracing::debug!(url, model = %config.model, "stream request");

        let response = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(LlmError::Http(status, body_text));
        }

        let mut usage = TokenUsage::default();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut accumulated_args: HashMap<String, String> = HashMap::new();

        let mut stream = response.bytes_stream();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(LlmError::Network)?;
            let text = String::from_utf8_lossy(&chunk).to_string();

            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() || line == "data: [DONE]" {
                    continue;
                }

                let json_str = if line.starts_with("data: ") {
                    &line[6..]
                } else {
                    line
                };

                let parsed: serde_json::Value =
                    serde_json::from_str(json_str).map_err(LlmError::Deserialize)?;

                let choices = parsed.get("choices");
                if let Some(choices) = choices {
                    for choice in choices.as_array().unwrap_or(&vec![]).iter() {
                        let delta = choice.get("delta");

                        if let Some(content) = delta.and_then(|d| d.get("content")) {
                            let text = content.as_str().unwrap_or("");
                            // Forward chunk to TUI for progressive rendering
                            if let Some(tx) = chunk_tx {
                                let _ = tx.send(text.to_string());
                            }
                            full_content.push_str(text);
                        }

                        if let Some(tc) = delta.and_then(|d| d.get("tool_calls")) {
                            for tc_item in tc.as_array().unwrap_or(&vec![]) {
                                let id = tc_item
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let name = tc_item
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let args = tc_item
                                    .get("function")
                                    .and_then(|f| f.get("arguments"))
                                    .and_then(|a| a.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                if !id.is_empty() {
                                    accumulated_args.entry(id.clone()).or_default();
                                    accumulated_args.get_mut(&id).unwrap().push_str(&args);

                                    if tool_calls.iter().all(|t| t.id != id) {
                                        tool_calls.push(ToolCall {
                                            id: id.clone(),
                                            name: name.clone(),
                                            arguments: serde_json::Value::Null,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }

                if is_anthropic {
                    if let Some(usage_val) = parsed.get("usage") {
                        let mut u = TokenUsage {
                            prompt_tokens: usage_val
                                .get("input_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as u32,
                            completion_tokens: usage_val
                                .get("output_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as u32,
                            total_tokens: 0,
                        };
                        u.total_tokens = u.prompt_tokens + u.completion_tokens;
                        usage.add(&u);
                    }
                } else {
                    if let Some(usage_val) = parsed.get("usage") {
                        let mut u = TokenUsage {
                            prompt_tokens: usage_val
                                .get("prompt_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as u32,
                            completion_tokens: usage_val
                                .get("completion_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as u32,
                            total_tokens: 0,
                        };
                        u.total_tokens = u.prompt_tokens + u.completion_tokens;
                        usage.add(&u);
                    }
                }
            }
        }

        for tc in tool_calls.iter_mut() {
            if let Some(args_str) = accumulated_args.get(&tc.id) {
                tc.arguments =
                    serde_json::from_str(args_str).unwrap_or(serde_json::Value::Null);
            }
        }

        Ok(StreamResult {
            content: full_content,
            tool_calls,
            usage,
        })
    }

    // -- Endpoint resolution -------------------------------------------------

    fn resolve_endpoint(&self, config: &LlmConfig) -> Result<(String, bool), LlmError> {
        let provider = config.provider.to_lowercase();

        let base_url = if let Some(ref url) = config.base_url {
            url.clone()
        } else {
            match provider.as_str() {
                "openai" => "https://api.openai.com/v1".into(),
                "openrouter" => "https://openrouter.ai/api/v1".into(),
                "anthropic" => "https://api.anthropic.com".into(),
                "lmstudio" => "http://localhost:1234/v1".into(),
                "ollama" => "http://localhost:11434/v1".into(),
                _ => "https://api.openai.com/v1".into(),
            }
        };

        let is_anthropic = provider == "anthropic";
        Ok((base_url, is_anthropic))
    }

    fn effective_key(&self, config: &LlmConfig) -> Result<String, LlmError> {
        std::env::var("POLYMDE_LLM_API_KEY")
            .ok()
            .or_else(|| config.api_key.clone())
            .ok_or(LlmError::NoApiKey)
    }

    // -- Request body builders (OpenAI format) --------------------------------

    fn build_openai_body(
        &self,
        config: &LlmConfig,
        messages: &[Message],
        tools: &Option<Vec<ToolDefinition>>,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::Map::new();
        body.insert("model".into(), serde_json::Value::String(config.model.clone()));
        body.insert(
            "messages".into(),
            serde_json::to_value(messages).map_err(LlmError::Deserialize)?,
        );
        body.insert(
            "temperature".into(),
            serde_json::Value::Number(serde_json::Number::from_f64(self.temperature as f64)
                .unwrap_or(serde_json::Number::from_f64(0.7).unwrap())),
        );

        if let Some(tokens) = self.max_tokens {
            body.insert("max_tokens".into(), serde_json::Value::Number(tokens.into()));
        }

        if let Some(tools) = tools {
            let openai_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    let mut tool = serde_json::Map::new();
                    tool.insert("type".into(), "function".into());
                    let mut func = serde_json::Map::new();
                    func.insert("name".into(), serde_json::Value::String(t.name.clone()));
                    func.insert(
                        "description".into(),
                        serde_json::Value::String(t.description.clone()),
                    );
                    func.insert("parameters".into(), t.schema.clone());
                    tool.insert("function".into(), serde_json::Value::Object(func));
                    serde_json::Value::Object(tool)
                })
                .collect();
            body.insert("tools".into(), serde_json::Value::Array(openai_tools));
            body.insert(
                "tool_choice".into(),
                serde_json::Value::String("auto".into()),
            );
        }

        Ok(serde_json::Value::Object(body))
    }

    fn build_openai_stream_body(
        &self,
        config: &LlmConfig,
        messages: &[Message],
        tools: &Option<Vec<ToolDefinition>>,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = self.build_openai_body(config, messages, tools)?;
        if let Some(obj) = body.as_object_mut() {
            obj.insert("stream".into(), serde_json::Value::Bool(true));
            obj.insert(
                "stream_options".into(),
                {
                    let mut so = serde_json::Map::new();
                    so.insert("include_usage".into(), serde_json::Value::Bool(true));
                    serde_json::Value::Object(so)
                },
            );
        }
        Ok(body)
    }

    // -- Request body builders (Anthropic format) -----------------------------

    fn build_anthropic_body(
        &self,
        config: &LlmConfig,
        messages: &[Message],
        tools: &Option<Vec<ToolDefinition>>,
        _api_key: &str,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::Map::new();
        body.insert("model".into(), serde_json::Value::String(config.model.clone()));

        let mut system_parts: Vec<String> = Vec::new();
        let mut anthropic_messages: Vec<serde_json::Value> = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    system_parts.push(msg.content.clone());
                }
                MessageRole::Assistant => {
                    let mut content_arr = Vec::new();
                    if !msg.content.is_empty() {
                        content_arr.push(serde_json::json!({
                            "type": "text",
                            "text": msg.content
                        }));
                    }
                    if let Some(ref calls) = msg.tool_calls {
                        for call in calls {
                            content_arr.push(serde_json::json!({
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.name,
                                "input": call.arguments
                            }));
                        }
                    }
                    anthropic_messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": content_arr
                    }));
                }
                MessageRole::Tool => {
                    anthropic_messages.push(serde_json::json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id.as_deref().unwrap_or(""),
                            "content": msg.content
                        }]
                    }));
                }
                MessageRole::User => {
                    anthropic_messages.push(serde_json::json!({
                        "role": "user",
                        "content": msg.content
                    }));
                }
            }
        }

        if !system_parts.is_empty() {
            body.insert("system".into(), serde_json::Value::String(system_parts.join("\n")));
        }

        body.insert(
            "messages".into(),
            serde_json::Value::Array(anthropic_messages),
        );
        body.insert(
            "max_tokens".into(),
            serde_json::Value::Number(
                self.max_tokens
                    .unwrap_or(8192)
                    .into(),
            ),
        );

        if let Some(t) = serde_json::Number::from_f64(self.temperature as f64) {
            body.insert("temperature".into(), serde_json::Value::Number(t));
        }

        if let Some(tools) = tools {
            let anthropic_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.schema
                    })
                })
                .collect();
            body.insert("tools".into(), serde_json::Value::Array(anthropic_tools));
        }

        let provider = config.provider.to_lowercase();
        if provider == "anthropic" {
            body.remove("anthropic_version");
            body.insert(
                "anthropic_version".into(),
                serde_json::Value::String("2023-06-01".into()),
            );
        }

        Ok(serde_json::Value::Object(body))
    }

    fn build_anthropic_stream_body(
        &self,
        config: &LlmConfig,
        messages: &[Message],
        tools: &Option<Vec<ToolDefinition>>,
        api_key: &str,
    ) -> Result<serde_json::Value, LlmError> {
        let mut body = self.build_anthropic_body(config, messages, tools, api_key)?;
        if let Some(obj) = body.as_object_mut() {
            obj.insert("stream".into(), serde_json::Value::Bool(true));
        }
        Ok(body)
    }

    // -- Response parsers ----------------------------------------------------

    async fn parse_openai_usage(
        &self,
        response: reqwest::Response,
    ) -> Result<ChatResponse, LlmError> {
        let json: serde_json::Value = response.json().await?;

        let content = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let tool_calls = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("tool_calls"))
            .and_then(|tc| tc.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|tc| {
                        let id = tc.get("id")?.as_str()?.to_string();
                        let name = tc.get("function")?
                            .get("name")?
                            .as_str()?
                            .to_string();
                        let args_str = tc.get("function")?
                            .get("arguments")?
                            .as_str()?
                            .to_string();
                        let arguments: serde_json::Value =
                            serde_json::from_str(&args_str).unwrap_or_default();
                        Some(ToolCall { id, name, arguments })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let usage = json.get("usage").map(|u| TokenUsage {
            prompt_tokens: u.get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: u.get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: u.get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        });

        let usage = usage.unwrap_or_else(|| TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        });

        Ok(ChatResponse {
            content,
            tool_calls,
            usage,
        })
    }

    async fn parse_anthropic_usage(
        &self,
        response: reqwest::Response,
    ) -> Result<ChatResponse, LlmError> {
        let json: serde_json::Value = response.json().await?;

        let mut content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        if let Some(content_block) = json.get("content") {
            for block in content_block.as_array().unwrap_or(&vec![]) {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match block_type {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            content.push_str(text);
                        }
                    }
                    "tool_use" => {
                        if let (Some(id), Some(name)) = (
                            block.get("id").and_then(|v| v.as_str()),
                            block.get("name").and_then(|v| v.as_str()),
                        ) {
                            let input = block.get("input").cloned().unwrap_or_default();
                            tool_calls.push(ToolCall {
                                id: id.to_string(),
                                name: name.to_string(),
                                arguments: input,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        let usage = json.get("usage").map(|u| TokenUsage {
            prompt_tokens: u.get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: u.get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: 0,
        });

        let usage = usage.unwrap_or_else(|| TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        });

        let total = usage.prompt_tokens + usage.completion_tokens;

        Ok(ChatResponse {
            content,
            tool_calls,
            usage: TokenUsage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: total,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResult {
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
}

impl From<StreamResult> for ChatResponse {
    fn from(s: StreamResult) -> Self {
        ChatResponse {
            content: s.content,
            tool_calls: s.tool_calls,
            usage: s.usage,
        }
    }
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

impl Default for TokenUsage {
    fn default() -> Self {
        TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_add() {
        let mut a = TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };
        let b = TokenUsage {
            prompt_tokens: 5,
            completion_tokens: 15,
            total_tokens: 20,
        };
        a.add(&b);
        assert_eq!(a.prompt_tokens, 15);
        assert_eq!(a.completion_tokens, 35);
        assert_eq!(a.total_tokens, 50);
    }

    #[test]
    fn endpoint_resolution_openai() {
        let client = LlmClient::new(LlmConfig {
            provider: "openai".into(),
            model: "gpt-4".into(),
            api_key: Some("sk-test".into()),
            base_url: None,
            fallback: None,
        });
        let (url, is_antic) = client
            .resolve_endpoint(&client.config)
            .expect("resolve");
        assert_eq!(url, "https://api.openai.com/v1");
        assert!(!is_antic);
    }

    #[test]
    fn endpoint_resolution_anthropic() {
        let client = LlmClient::new(LlmConfig {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-20250514".into(),
            api_key: Some("sk-ant".into()),
            base_url: None,
            fallback: None,
        });
        let (url, is_antic) = client
            .resolve_endpoint(&client.config)
            .expect("resolve");
        assert_eq!(url, "https://api.anthropic.com");
        assert!(is_antic);
    }

    #[test]
    fn endpoint_resolution_custom_url() {
        let client = LlmClient::new(LlmConfig {
            provider: "custom".into(),
            model: "some-model".into(),
            api_key: Some("key".into()),
            base_url: Some("http://localhost:9999/v1".into()),
            fallback: None,
        });
        let (url, is_antic) = client
            .resolve_endpoint(&client.config)
            .expect("resolve");
        assert_eq!(url, "http://localhost:9999/v1");
        assert!(!is_antic);
    }

    #[test]
    fn effective_key_env_override() {
        let client = LlmClient::new(LlmConfig {
            provider: "openai".into(),
            model: "gpt-4".into(),
            api_key: Some("file-key".into()),
            base_url: None,
            fallback: None,
        });
        unsafe { std::env::set_var("POLYMDE_LLM_API_KEY", "env-key") };
        let key = client.effective_key(&client.config).expect("key");
        assert_eq!(key, "env-key");
        unsafe { std::env::remove_var("POLYMDE_LLM_API_KEY") };
    }

    #[test]
    fn effective_key_from_config() {
        let client = LlmClient::new(LlmConfig {
            provider: "openai".into(),
            model: "gpt-4".into(),
            api_key: Some("file-key".into()),
            base_url: None,
            fallback: None,
        });
        unsafe { std::env::remove_var("POLYMDE_LLM_API_KEY") };
        let key = client.effective_key(&client.config).expect("key");
        assert_eq!(key, "file-key");
    }

    #[test]
    fn effective_key_missing() {
        let client = LlmClient::new(LlmConfig {
            provider: "openai".into(),
            model: "gpt-4".into(),
            api_key: None,
            base_url: None,
            fallback: None,
        });
        unsafe { std::env::remove_var("POLYMDE_LLM_API_KEY") };
        let result = client.effective_key(&client.config);
        assert!(matches!(result, Err(LlmError::NoApiKey)));
    }

    #[test]
    fn openai_body_has_stream_flag() {
        let client = LlmClient::new(LlmConfig {
            provider: "openai".into(),
            model: "gpt-4".into(),
            api_key: Some("sk-test".into()),
            base_url: None,
            fallback: None,
        });
        let messages = vec![Message {
            role: MessageRole::User,
            content: "hello".into(),
            tool_calls: None,
            tool_call_id: None,
        }];
        let body = client
            .build_openai_stream_body(&client.config, &messages, &None)
            .expect("build");
        assert!(body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false));
        assert!(body.get("stream_options").is_some());
    }

    #[test]
    fn openai_body_includes_tools() {
        let client = LlmClient::new(LlmConfig {
            provider: "openai".into(),
            model: "gpt-4".into(),
            api_key: Some("sk-test".into()),
            base_url: None,
            fallback: None,
        });
        let messages = vec![Message {
            role: MessageRole::User,
            content: "hello".into(),
            tool_calls: None,
            tool_call_id: None,
        }];
        let tools = vec![ToolDefinition {
            name: "search".into(),
            description: "Search the web".into(),
            schema: serde_json::json!({"type": "object", "properties": {}}),
        }];
        let body = client
            .build_openai_body(&client.config, &messages, &Some(tools))
            .expect("build");
        assert!(body.get("tools").is_some());
        assert!(body.get("tool_choice").is_some());
    }

    #[test]
    fn anthropic_body_system_extraction() {
        let client = LlmClient::new(LlmConfig {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-20250514".into(),
            api_key: Some("sk-ant".into()),
            base_url: None,
            fallback: None,
        });
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are helpful".into(),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: MessageRole::User,
                content: "hi".into(),
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        let body = client
            .build_anthropic_body(&client.config, &messages, &None, "sk-ant")
            .expect("build");
        assert!(body.get("system").is_some());
        assert!(body.get("anthropic_version").is_some());
    }

    #[test]
    fn lmstudio_endpoint() {
        let client = LlmClient::new(LlmConfig {
            provider: "lmstudio".into(),
            model: "qwen3-27b".into(),
            api_key: None,
            base_url: None,
            fallback: None,
        });
        let (url, is_antic) = client
            .resolve_endpoint(&client.config)
            .expect("resolve");
        assert_eq!(url, "http://localhost:1234/v1");
        assert!(!is_antic);
    }

    #[test]
    fn ollama_endpoint() {
        let client = LlmClient::new(LlmConfig {
            provider: "ollama".into(),
            model: "llama3".into(),
            api_key: None,
            base_url: None,
            fallback: None,
        });
        let (url, is_antic) = client
            .resolve_endpoint(&client.config)
            .expect("resolve");
        assert_eq!(url, "http://localhost:11434/v1");
        assert!(!is_antic);
    }

    #[test]
    fn message_serialization() {
        let msg = Message {
            role: MessageRole::Assistant,
            content: "Hello".into(),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        let parsed: Message = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.role, MessageRole::Assistant);
        assert_eq!(parsed.content, "Hello");
    }
}
