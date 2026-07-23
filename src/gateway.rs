use std::collections::HashMap;
use std::sync::{atomic::AtomicBool, Arc};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

use crate::agent::{AgentInput, AgentOutput};

// ---------------------------------------------------------------------------
// Platform identity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    Telegram,
    Discord,
    Slack,
    WhatsApp,
    Signal,
    Email,
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Platform::Telegram => write!(f, "telegram"),
            Platform::Discord => write!(f, "discord"),
            Platform::Slack => write!(f, "slack"),
            Platform::WhatsApp => write!(f, "whatsapp"),
            Platform::Signal => write!(f, "signal"),
            Platform::Email => write!(f, "email"),
        }
    }
}

/// Unique identifier for a user on a given platform.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlatformUser {
    pub platform: Platform,
    pub user_id: String,
}

impl std::fmt::Display for PlatformUser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.platform, self.user_id)
    }
}

// ---------------------------------------------------------------------------
// Gateway configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub telegram: Option<TelegramConfig>,
    pub discord: Option<DiscordConfig>,
    pub slack: Option<SlackConfig>,
    pub whatsapp: Option<WhatsAppConfig>,
    pub signal: Option<SignalConfig>,
    pub email: Option<EmailConfig>,
    /// Poll interval for long-polling adapters (default 5 s).
    #[serde(default = "default_poll_interval_secs")]
    pub poll_interval_secs: u64,
    /// Reconnection backoff strategy configuration.
    #[serde(default)]
    pub reconnect: Option<ReconnectConfig>,
}

/// Configurable reconnection backoff strategy for gateway adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconnectConfig {
    /// Initial backoff duration in seconds (default 2).
    #[serde(default = "default_initial_backoff_secs")]
    pub initial_backoff_secs: u64,
    /// Maximum backoff duration in seconds (default 120).
    #[serde(default = "default_max_backoff_secs")]
    pub max_backoff_secs: u64,
    /// Multiplier for exponential growth (default 2.0).
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,
    /// Maximum number of reconnect attempts before giving up (default 10).
    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,
    /// Add random jitter to backoff (0.0–1.0 fraction, default 0.25).
    #[serde(default = "default_jitter_fraction")]
    pub jitter_fraction: f64,
}

fn default_initial_backoff_secs() -> u64 { 2 }
fn default_max_backoff_secs() -> u64 { 120 }
fn default_backoff_multiplier() -> f64 { 2.0 }
fn default_max_attempts() -> u32 { 10 }
fn default_jitter_fraction() -> f64 { 0.25 }

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            initial_backoff_secs: 2,
            max_backoff_secs: 120,
            backoff_multiplier: 2.0,
            max_attempts: 10,
            jitter_fraction: 0.25,
        }
    }
}

impl ReconnectConfig {
    /// Compute the next backoff duration with optional jitter.
    pub fn compute_backoff(&self, current: Duration, rng_seed: u64) -> Duration {
        let base = (current.as_secs_f64() * self.backoff_multiplier)
            .min(self.max_backoff_secs as f64);
        // Deterministic jitter based on seed to avoid needing a real RNG.
        let jitter_range = base * self.jitter_fraction;
        let jitter = ((rng_seed as f64 * 0.157319) % 1.0) * jitter_range;
        Duration::from_secs_f64((base + jitter).min(self.max_backoff_secs as f64))
    }
}

fn default_poll_interval_secs() -> u64 {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramConfig {
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordConfig {
    pub bot_token: String,
    pub application_id: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    pub bot_token: String,
    pub app_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatsAppConfig {
    pub api_key: String,
    pub phone_number_id: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    pub recipient_number: String,
    pub gateway_url: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    pub imap_host: String,
    pub imap_port: u16,
    pub imap_user: String,
    pub imap_password: String,
    pub smtp_host: String,
    pub smtp_port: u16,
    pub smtp_user: String,
    pub smtp_password: String,
    #[serde(default)]
    pub allowed_senders: Vec<String>,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            telegram: None,
            discord: None,
            slack: None,
            whatsapp: None,
            signal: None,
            email: None,
            poll_interval_secs: 5,
            reconnect: Some(ReconnectConfig::default()),
        }
    }
}

// ---------------------------------------------------------------------------
// Slash commands
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum SlashCommand {
    Help,
    Reset,
    Status,
    Model,
    Tools,
    Custom(String, Vec<String>),
}

impl SlashCommand {
    pub fn name(&self) -> &str {
        match self {
            SlashCommand::Help => "help",
            SlashCommand::Reset => "reset",
            SlashCommand::Status => "status",
            SlashCommand::Model => "model",
            SlashCommand::Tools => "tools",
            SlashCommand::Custom(name, _) => name,
        }
    }
}

pub fn parse_slash_command(input: &str) -> Option<(SlashCommand, String)> {
    let trimmed = input.trim();

    if !trimmed.starts_with('/') {
        return None;
    }

    let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
    let cmd = parts[0].trim_start_matches('/').to_lowercase();
    let rest = parts.get(1).map(|s| s.trim()).unwrap_or("").to_string();

    let slash = match cmd.as_str() {
        "help" => SlashCommand::Help,
        "reset" => SlashCommand::Reset,
        "status" => SlashCommand::Status,
        "model" => SlashCommand::Model,
        "tools" => SlashCommand::Tools,
        other => SlashCommand::Custom(
            other.to_string(),
            rest.split_whitespace().map(|s| s.to_string()).collect(),
        ),
    };

    Some((slash, rest))
}

// ---------------------------------------------------------------------------
// Conversation state
// ---------------------------------------------------------------------------

/// Tracks per-user conversation metadata across platforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationState {
    pub conversation_id: String,
    pub user: PlatformUser,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_active: chrono::DateTime<chrono::Utc>,
    pub message_count: u32,
}

impl ConversationState {
    pub fn new(user: PlatformUser) -> Self {
        Self {
            conversation_id: Uuid::new_v4().to_string(),
            user,
            created_at: chrono::Utc::now(),
            last_active: chrono::Utc::now(),
            message_count: 0,
        }
    }

    pub fn touch(&mut self) {
        self.last_active = chrono::Utc::now();
        self.message_count += 1;
    }
}

// ---------------------------------------------------------------------------
// Platform adapter trait
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
pub trait PlatformAdapter: Send + Sync {
    /// Human-readable platform name.
    fn platform_name(&self) -> Platform;

    /// Start listening for incoming messages.
    /// Blocks until an error occurs or shutdown is requested.
    /// The adapter sends messages into `inbox_tx` for the gateway to process.
    async fn start_listening(
        &self,
        inbox_tx: mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError>;

    /// Send a response back to the originating user.
    async fn send_message(&self, user_id: &str, content: &str) -> Result<(), GatewayError>;

    /// Shut down the adapter gracefully.
    async fn shutdown(&self);

    /// Check if a user is authorized on this platform.
    fn is_authorized(&self, user_id: &str) -> bool;
}

// ---------------------------------------------------------------------------
// Inbound message envelope
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct InboundMessage {
    pub platform: Platform,
    pub user_id: String,
    pub content: String,
    /// Optional thread/topic identifier (Telegram forum topic ID, Discord channel/thread ID).
    pub thread_id: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// ---------------------------------------------------------------------------
// Gateway error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum GatewayError {
    Connection(String),
    Send(String),
    Auth(String),
    Config(String),
    Shutdown,
}

impl std::fmt::Display for GatewayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GatewayError::Connection(msg) => write!(f, "connection error: {msg}"),
            GatewayError::Send(msg) => write!(f, "send error: {msg}"),
            GatewayError::Auth(msg) => write!(f, "auth error: {msg}"),
            GatewayError::Config(msg) => write!(f, "config error: {msg}"),
            GatewayError::Shutdown => write!(f, "gateway shutting down"),
        }
    }
}

impl std::error::Error for GatewayError {}

// ---------------------------------------------------------------------------
// Telegram adapter — Bot API long-polling via getUpdates
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TelegramAdapter {
    config: TelegramConfig,
    client: reqwest::Client,
    shutdown_flag: Arc<AtomicBool>,
}

impl TelegramAdapter {
    pub fn new(config: TelegramConfig) -> Self {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        Self {
            config,
            client: reqwest::Client::new(),
            shutdown_flag,
        }
    }

    /// Poll Telegram Bot API for new updates.
    async fn poll_updates(
        &self,
        inbox_tx: &mpsc::UnboundedSender<InboundMessage>,
        last_update_id: i64,
    ) -> Result<i64, GatewayError> {
        let url = format!(
            "https://api.telegram.org/bot{}/getUpdates",
            self.config.bot_token
        );
        let params = [
            ("timeout", "25"),
            ("offset", &(last_update_id + 1).to_string()),
            ("limit", "100"),
        ];

        let resp = match self.client.get(&url).query(&params).send().await {
            Ok(r) => r,
            Err(e) => return Err(GatewayError::Connection(format!("Telegram API request failed: {e}"))),
        };

        if !resp.status().is_success() {
            let status = resp.status();
            return Err(GatewayError::Connection(format!(
                "Telegram API returned {}",
                status
            )));
        }

        let body: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return Err(GatewayError::Connection(format!("Failed to parse Telegram response: {e}"))),
        };

        let empty: Vec<serde_json::Value> = vec![];
        let updates = body.get("result").and_then(|r| r.as_array()).unwrap_or(&empty);
        let mut highest_id = last_update_id;

        for update in updates {
            let update_id = update.get("update_id").and_then(|v| v.as_i64()).unwrap_or(0);
            if update_id > highest_id {
                highest_id = update_id;
            }

            // Extract message content from the update.
            if let Some(message) = update.get("message") {
                let text = message.get("text").and_then(|t| t.as_str());
                let chat_id = message.get("chat").and_then(|c| c.get("id")).map(|i| i.to_string());
                let from_id = message.get("from").and_then(|f| f.get("id")).map(|i| i.to_string());

                if let (Some(content), Some(user_id)) = (text, from_id) {
                    tracing::debug!(
                        platform = "telegram",
                        user = %user_id,
                        chat = %chat_id.unwrap_or_default(),
                        "incoming message"
                    );

                    let _ = inbox_tx.send(InboundMessage {
                        platform: Platform::Telegram,
                        user_id,
                        content: content.to_string(),
                        thread_id: None, // Telegram topics wired in future adapter update.
                        timestamp: chrono::Utc::now(),
                    });
                }
            }
        }

        Ok(highest_id)
    }
}

#[async_trait::async_trait]
impl PlatformAdapter for TelegramAdapter {
    fn platform_name(&self) -> Platform {
        Platform::Telegram
    }

    async fn start_listening(
        &self,
        inbox_tx: mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        tracing::info!(platform = "telegram", "adapter started — long-polling getUpdates");

        let mut last_update_id: i64 = -1;

        loop {
            if self.shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(platform = "telegram", "shutdown requested");
                return Err(GatewayError::Shutdown);
            }
            match self.poll_updates(&inbox_tx, last_update_id).await {
                Ok(id) => last_update_id = id,
                Err(e) => {
                    tracing::warn!(platform = "telegram", error = %e, "poll failed");
                    return Err(e);
                }
            }
        }
    }

    async fn send_message(&self, user_id: &str, content: &str) -> Result<(), GatewayError> {
        let url = format!(
            "https://api.telegram.org/bot{}/sendMessage",
            self.config.bot_token
        );

        let resp = match self.client.post(&url).json(&serde_json::json!({
            "chat_id": user_id,
            "text": content,
        })).send().await {
            Ok(r) => r,
            Err(e) => return Err(GatewayError::Send(format!("Telegram send request failed: {e}"))),
        };

        if !resp.status().is_success() {
            return Err(GatewayError::Send(format!(
                "Telegram send returned {}",
                resp.status()
            )));
        }

        tracing::debug!(platform = "telegram", user = %user_id, "message sent");
        Ok(())
    }

    async fn shutdown(&self) {
        tracing::info!(platform = "telegram", "adapter shutting down");
    }

    fn is_authorized(&self, user_id: &str) -> bool {
        if self.config.allowed_user_ids.is_empty() {
            return true;
        }
        self.config.allowed_user_ids.contains(&user_id.to_string())
    }
}

// ---------------------------------------------------------------------------
// Discord adapter — Gateway Intent polling (REST fallback)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct DiscordAdapter {
    config: DiscordConfig,
    client: reqwest::Client,
    shutdown_flag: Arc<AtomicBool>,
}

impl DiscordAdapter {
    pub fn new(config: DiscordConfig) -> Self {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        Self {
            config,
            client: reqwest::Client::new(),
            shutdown_flag,
        }
    }

    /// In production this would connect to the Discord Gateway WebSocket.
    /// As a REST-based polling fallback we check for new interactions / messages.
    async fn poll_messages(
        &self,
        _inbox_tx: &mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        // Production note: Discord's real-time messaging requires the Gateway WebSocket
        // protocol (identify → heartbeat → dispatch events). A REST-only approach can
        // only read channel messages if we know the channel IDs ahead of time.
        // This stub logs the intent and would be replaced by a tokio_tungstenite
        // connection in production.

        tracing::trace!(platform = "discord", "poll cycle (gateway WebSocket not connected)");

        // If interaction endpoints were configured, we could poll them here:
        // GET /channels/{channel_id}/messages?after={message_id}
        // For now this is a no-op poll that keeps the loop alive.
        Ok(())
    }
}

#[async_trait::async_trait]
impl PlatformAdapter for DiscordAdapter {
    fn platform_name(&self) -> Platform {
        Platform::Discord
    }

    async fn start_listening(
        &self,
        inbox_tx: mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        tracing::info!(
            platform = "discord",
            application_id = %self.config.application_id,
            "adapter started — awaiting gateway WebSocket connection"
        );

        loop {
            if self.shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(platform = "discord", "shutdown requested");
                return Err(GatewayError::Shutdown);
            }
            if let Err(e) = self.poll_messages(&inbox_tx).await {
                tracing::warn!(platform = "discord", error = %e, "poll failed");
                return Err(e);
            }
        }
    }

    async fn send_message(&self, user_id: &str, content: &str) -> Result<(), GatewayError> {
        // In production this would use the Discord REST API:
        // POST /channels/{channel_id}/messages with body {"content": ...}
        // The user_id here is treated as a channel ID for DMs.
        let url = format!("https://discord.com/api/v9/channels/{}/messages", user_id);

        let resp = match self.client.post(&url)
            .header("Authorization", format!("Bot {}", self.config.bot_token))
            .json(&serde_json::json!({ "content": content }))
            .send().await
        {
            Ok(r) => r,
            Err(e) => return Err(GatewayError::Send(format!("Discord send request failed: {e}"))),
        };

        if !resp.status().is_success() {
            return Err(GatewayError::Send(format!(
                "Discord send returned {}",
                resp.status()
            )));
        }

        tracing::debug!(platform = "discord", user = %user_id, "message sent");
        Ok(())
    }

    async fn shutdown(&self) {
        tracing::info!(platform = "discord", "adapter shutting down");
    }

    fn is_authorized(&self, user_id: &str) -> bool {
        if self.config.allowed_user_ids.is_empty() {
            return true;
        }
        self.config.allowed_user_ids.contains(&user_id.to_string())
    }
}

// ---------------------------------------------------------------------------
// Slack adapter — Events API / RTM polling
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct SlackAdapter {
    config: SlackConfig,
    client: reqwest::Client,
    shutdown_flag: Arc<AtomicBool>,
}

impl SlackAdapter {
    pub fn new(config: SlackConfig) -> Self {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        Self {
            config,
            client: reqwest::Client::new(),
            shutdown_flag,
        }
    }

    /// Poll Slack for new messages via conversations.history.
    /// In production this would be replaced by an Events API webhook listener
    /// or the RTM connect WebSocket.
    async fn poll_channels(
        &self,
        _inbox_tx: &mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        // Production note: Slack's recommended approach is the Events API with HTTP
        // callbacks (message.im / message.channel). A polling fallback uses:
        //   1. conversations.list → get channel IDs
        //   2. conversations.history per channel → new messages since last cursor
        // This stub keeps the poll loop alive for the reconnect wrapper.

        tracing::trace!(platform = "slack", "poll cycle (Events API webhook not configured)");
        Ok(())
    }
}

#[async_trait::async_trait]
impl PlatformAdapter for SlackAdapter {
    fn platform_name(&self) -> Platform {
        Platform::Slack
    }

    async fn start_listening(
        &self,
        inbox_tx: mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        tracing::info!(platform = "slack", "adapter started — awaiting Events API / RTM");

        loop {
            if self.shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(platform = "slack", "shutdown requested");
                return Err(GatewayError::Shutdown);
            }
            if let Err(e) = self.poll_channels(&inbox_tx).await {
                tracing::warn!(platform = "slack", error = %e, "poll failed");
                return Err(e);
            }
        }
    }

    async fn send_message(&self, user_id: &str, content: &str) -> Result<(), GatewayError> {
        // Slack chat.postMessage API
        let url = "https://slack.com/api/chat.postMessage";

        let resp = match self.client.post(url)
            .bearer_auth(&self.config.bot_token)
            .form(&[("channel", user_id), ("text", content)])
            .send().await
        {
            Ok(r) => r,
            Err(e) => return Err(GatewayError::Send(format!("Slack send request failed: {e}"))),
        };

        if !resp.status().is_success() {
            return Err(GatewayError::Send(format!(
                "Slack send returned {}",
                resp.status()
            )));
        }

        tracing::debug!(platform = "slack", user = %user_id, "message sent");
        Ok(())
    }

    async fn shutdown(&self) {
        tracing::info!(platform = "slack", "adapter shutting down");
    }

    fn is_authorized(&self, user_id: &str) -> bool {
        if self.config.allowed_user_ids.is_empty() {
            return true;
        }
        self.config.allowed_user_ids.contains(&user_id.to_string())
    }
}

// ---------------------------------------------------------------------------
// WhatsApp adapter — Cloud API (Meta Graph API)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct WhatsAppAdapter {
    config: WhatsAppConfig,
    client: reqwest::Client,
    shutdown_flag: Arc<AtomicBool>,
}

impl WhatsAppAdapter {
    pub fn new(config: WhatsAppConfig) -> Self {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        Self {
            config,
            client: reqwest::Client::new(),
            shutdown_flag,
        }
    }

    /// Poll WhatsApp Cloud API for messages.
    /// In production this would use a webhook callback URL registered with Meta.
    async fn poll_messages(
        &self,
        _inbox_tx: &mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        // Production note: WhatsApp Business Cloud API delivers inbound messages via
        // webhook callbacks to a URL you configure in the Meta developer console.
        // There is no polling endpoint for new messages. This stub keeps the loop alive.

        tracing::trace!(platform = "whatsapp", "poll cycle (webhook not configured)");
        Ok(())
    }
}

#[async_trait::async_trait]
impl PlatformAdapter for WhatsAppAdapter {
    fn platform_name(&self) -> Platform {
        Platform::WhatsApp
    }

    async fn start_listening(
        &self,
        inbox_tx: mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        tracing::info!(
            platform = "whatsapp",
            phone_number_id = %self.config.phone_number_id,
            "adapter started — awaiting webhook delivery"
        );

        loop {
            if self.shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(platform = "whatsapp", "shutdown requested");
                return Err(GatewayError::Shutdown);
            }
            if let Err(e) = self.poll_messages(&inbox_tx).await {
                tracing::warn!(platform = "whatsapp", error = %e, "poll failed");
                return Err(e);
            }
        }
    }

    async fn send_message(&self, user_id: &str, content: &str) -> Result<(), GatewayError> {
        // WhatsApp Cloud API — POST to /{phone_number_id}/messages
        let url = format!(
            "https://graph.facebook.com/v18.0/{}/messages",
            self.config.phone_number_id
        );

        let resp = match self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&serde_json::json!({
                "messaging_product": "whatsapp",
                "to": user_id,
                "type": "text",
                "text": { "body": content },
            }))
            .send().await
        {
            Ok(r) => r,
            Err(e) => return Err(GatewayError::Send(format!("WhatsApp send request failed: {e}"))),
        };

        if !resp.status().is_success() {
            return Err(GatewayError::Send(format!(
                "WhatsApp send returned {}",
                resp.status()
            )));
        }

        tracing::debug!(platform = "whatsapp", user = %user_id, "message sent");
        Ok(())
    }

    async fn shutdown(&self) {
        tracing::info!(platform = "whatsapp", "adapter shutting down");
    }

    fn is_authorized(&self, user_id: &str) -> bool {
        if self.config.allowed_user_ids.is_empty() {
            return true;
        }
        self.config.allowed_user_ids.contains(&user_id.to_string())
    }
}

// ---------------------------------------------------------------------------
// Signal adapter — signal-cli REST API gateway
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct SignalAdapter {
    config: SignalConfig,
    client: reqwest::Client,
    shutdown_flag: Arc<AtomicBool>,
}

impl SignalAdapter {
    pub fn new(config: SignalConfig) -> Self {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        Self {
            config,
            client: reqwest::Client::new(),
            shutdown_flag,
        }
    }

    /// Poll signal-cli REST API for new messages in the inbox.
    async fn poll_inbox(
        &self,
        inbox_tx: &mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        let url = format!("{}/inbox", self.config.gateway_url);

        let resp = match self.client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => return Err(GatewayError::Connection(format!("Signal gateway request failed: {e}"))),
        };

        if !resp.status().is_success() {
            // 404 / empty inbox is normal — don't treat it as a fatal error.
            if resp.status().as_u16() == 404 {
                return Ok(());
            }
            return Err(GatewayError::Connection(format!(
                "Signal gateway returned {}",
                resp.status()
            )));
        }

        let messages: Vec<serde_json::Value> = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(platform = "signal", error = %e, "failed to parse inbox");
                return Ok(());
            }
        };

        for msg in messages {
            let from = msg.get("from").and_then(|f| f.as_str()).unwrap_or("");
            let body = msg.get("body").and_then(|b| b.as_str()).unwrap_or("");
            if !body.is_empty() {
                tracing::debug!(platform = "signal", user = %from, "incoming message");

                let _ = inbox_tx.send(InboundMessage {
                    platform: Platform::Signal,
                    user_id: from.to_string(),
                    content: body.to_string(),
                    thread_id: None, // Signal threads wired in future adapter update.
                    timestamp: chrono::Utc::now(),
                });
            }
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl PlatformAdapter for SignalAdapter {
    fn platform_name(&self) -> Platform {
        Platform::Signal
    }

    async fn start_listening(
        &self,
        inbox_tx: mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        tracing::info!(
            platform = "signal",
            gateway_url = %self.config.gateway_url,
            "adapter started — polling signal-cli REST API"
        );

        loop {
            if self.shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(platform = "signal", "shutdown requested");
                return Err(GatewayError::Shutdown);
            }
            if let Err(e) = self.poll_inbox(&inbox_tx).await {
                tracing::warn!(platform = "signal", error = %e, "poll failed");
                return Err(e);
            }
        }
    }

    async fn send_message(&self, user_id: &str, content: &str) -> Result<(), GatewayError> {
        // signal-cli REST API — POST /send with recipient and message
        let url = format!("{}/send", self.config.gateway_url);

        let resp = match self.client.post(&url)
            .json(&serde_json::json!({
                "recipients": [user_id],
                "message": content,
            }))
            .send().await
        {
            Ok(r) => r,
            Err(e) => return Err(GatewayError::Send(format!("Signal send request failed: {e}"))),
        };

        if !resp.status().is_success() {
            return Err(GatewayError::Send(format!(
                "Signal send returned {}",
                resp.status()
            )));
        }

        tracing::debug!(platform = "signal", user = %user_id, "message sent");
        Ok(())
    }

    async fn shutdown(&self) {
        tracing::info!(platform = "signal", "adapter shutting down");
    }

    fn is_authorized(&self, user_id: &str) -> bool {
        if self.config.allowed_user_ids.is_empty() {
            return true;
        }
        self.config.allowed_user_ids.contains(&user_id.to_string())
    }
}

// ---------------------------------------------------------------------------
// Email adapter — IMAP polling + SMTP sending
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct EmailAdapter {
    config: EmailConfig,
    client: reqwest::Client,
    shutdown_flag: Arc<AtomicBool>,
}

impl EmailAdapter {
    pub fn new(config: EmailConfig) -> Self {
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        Self {
            config,
            client: reqwest::Client::new(),
            shutdown_flag,
        }
    }

    /// Poll IMAP for new unread messages.
    /// In production this would use a proper IMAP TLS client (e.g., async-imap).
    /// Here we demonstrate the polling structure with an HTTP-based proxy endpoint
    /// that wraps IMAP access.
    async fn poll_imap(
        &self,
        _inbox_tx: &mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        // Production note: Real IMAP polling requires an async IMAP client library
        // (e.g., `async-imap` or `lettre` for SMTP). The flow would be:
        //   1. Connect to imap_host:imap_port with TLS
        //   2. LOGIN with imap_user / imap_password
        //   3. SELECT INBOX
        //   4. SEARCH UNSEEN
        //   5. FETCH each new message body + headers
        //   6. Mark as SEEN
        // This stub keeps the poll loop alive for the reconnect wrapper.

        tracing::trace!(platform = "email", "poll cycle (IMAP client not connected)");
        Ok(())
    }
}

#[async_trait::async_trait]
impl PlatformAdapter for EmailAdapter {
    fn platform_name(&self) -> Platform {
        Platform::Email
    }

    async fn start_listening(
        &self,
        inbox_tx: mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), GatewayError> {
        tracing::info!(
            platform = "email",
            imap_host = %self.config.imap_host,
            imap_port = self.config.imap_port,
            "adapter started — awaiting IMAP connection"
        );

        loop {
            if self.shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(platform = "email", "shutdown requested");
                return Err(GatewayError::Shutdown);
            }
            if let Err(e) = self.poll_imap(&inbox_tx).await {
                tracing::warn!(platform = "email", error = %e, "poll failed");
                return Err(e);
            }
        }
    }

    async fn send_message(&self, user_id: &str, content: &str) -> Result<(), GatewayError> {
        // In production this would use lettre to send via SMTP.
        // As a stub we construct the correct SMTP submission parameters and log them.
        tracing::info!(
            platform = "email",
            smtp_host = %self.config.smtp_host,
            smtp_port = self.config.smtp_port,
            to = %user_id,
            "would send email via SMTP"
        );

        // Build a proper MIME message for logging / future integration:
        let _mime_msg = format!(
            "From: {}\r\nTo: {}\r\nSubject: Polymede Response\r\n\r\n{}",
            self.config.smtp_user, user_id, content
        );

        Ok(())
    }

    async fn shutdown(&self) {
        tracing::info!(platform = "email", "adapter shutting down");
    }

    fn is_authorized(&self, user_id: &str) -> bool {
        if self.config.allowed_senders.is_empty() {
            return true;
        }
        self.config.allowed_senders.contains(&user_id.to_string())
    }
}

// ---------------------------------------------------------------------------
// Gateway
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct Gateway {
    config: GatewayConfig,
    adapters: Vec<Arc<dyn PlatformAdapter>>,
    conversations: Arc<RwLock<HashMap<PlatformUser, ConversationState>>>,
    agent_input_tx: mpsc::UnboundedSender<AgentInput>,
    agent_output_rx: mpsc::UnboundedReceiver<AgentOutput>,
    shutdown_flag: Arc<AtomicBool>,
}

impl Gateway {
    pub fn new(
        config: GatewayConfig,
        agent_input_tx: mpsc::UnboundedSender<AgentInput>,
        agent_output_rx: mpsc::UnboundedReceiver<AgentOutput>,
    ) -> Self {
        let shutdown_flag = Arc::new(AtomicBool::new(false));

        let adapters = Self::build_adapters(&config);

        Self {
            config,
            adapters,
            conversations: Arc::new(RwLock::new(HashMap::new())),
            agent_input_tx,
            agent_output_rx,
            shutdown_flag,
        }
    }

    fn build_adapters(config: &GatewayConfig) -> Vec<Arc<dyn PlatformAdapter>> {
        let mut adapters: Vec<Arc<dyn PlatformAdapter>> = Vec::new();

        if let Some(cfg) = &config.telegram {
            adapters.push(Arc::new(TelegramAdapter::new(cfg.clone())));
        }
        if let Some(cfg) = &config.discord {
            adapters.push(Arc::new(DiscordAdapter::new(cfg.clone())));
        }
        if let Some(cfg) = &config.slack {
            adapters.push(Arc::new(SlackAdapter::new(cfg.clone())));
        }
        if let Some(cfg) = &config.whatsapp {
            adapters.push(Arc::new(WhatsAppAdapter::new(cfg.clone())));
        }
        if let Some(cfg) = &config.signal {
            adapters.push(Arc::new(SignalAdapter::new(cfg.clone())));
        }
        if let Some(cfg) = &config.email {
            adapters.push(Arc::new(EmailAdapter::new(cfg.clone())));
        }

        adapters
    }

    /// Start the gateway event loop.
    pub async fn run(mut self) -> Result<(), GatewayError> {
        tracing::info!(
            adapters = self.adapters.len(),
            poll_interval_secs = self.config.poll_interval_secs,
            "gateway starting"
        );

        // Start all platform adapters.
        let (inbox_tx, mut inbox_rx) = mpsc::unbounded_channel::<InboundMessage>();
        let reconnect_config = self.config.reconnect.clone().unwrap_or_default();

        let adapter_clones: Vec<Arc<dyn PlatformAdapter>> = self.adapters.clone();
        let mut adapter_handles = Vec::new();
        for adapter in adapter_clones {
            let name = adapter.platform_name();
            let tx = inbox_tx.clone();
            let rc = reconnect_config.clone();

            let handle = tokio::spawn(async move {
                Self::run_with_reconnect(adapter, tx, rc).await;
                tracing::info!(platform = %name, "adapter exited");
            });

            adapter_handles.push((name, handle));
        }

        // Main event loop: route inbound messages to agent, route agent output back.
        loop {
            if self.shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!("gateway shutdown requested");
                break;
            }

            tokio::select! {
                biased;

                maybe_msg = inbox_rx.recv() => {
                    match maybe_msg {
                        Some(msg) => {
                            if let Err(e) = self.route_inbound(msg).await {
                                tracing::error!(error = %e, "failed to route inbound message");
                            }
                        }
                        None => {
                            tracing::warn!("inbox channel closed");
                            break;
                        }
                    }
                }

                maybe_output = self.agent_output_rx.recv() => {
                    match maybe_output {
                        Some(output) => {
                            if let Err(e) = self.route_outbound(output).await {
                                tracing::error!(error = %e, "failed to route outbound message");
                            }
                        }
                        None => {
                            tracing::warn!("agent output channel closed");
                            break;
                        }
                    }
                }
            }
        }

        // Graceful shutdown: stop all adapters.
        for adapter in &self.adapters {
            adapter.shutdown().await;
        }

        for (name, handle) in adapter_handles {
            if let Err(e) = handle.await {
                tracing::error!(platform = %name, error = %e, "adapter join error");
            }
        }

        tracing::info!("gateway stopped");
        Ok(())
    }

    /// Run an adapter with automatic reconnection on failure.
    async fn run_with_reconnect(
        adapter: Arc<dyn PlatformAdapter>,
        inbox_tx: mpsc::UnboundedSender<InboundMessage>,
        config: ReconnectConfig,
    ) {
        let platform = adapter.platform_name();
        let mut backoff = Duration::from_secs(config.initial_backoff_secs);
        let max_backoff = Duration::from_secs(config.max_backoff_secs);
        let mut attempts = 0u32;

        loop {
            match adapter.start_listening(inbox_tx.clone()).await {
                Ok(()) => {
                    tracing::info!(platform = %platform, "adapter connected");
                    attempts = 0;
                    backoff = Duration::from_secs(config.initial_backoff_secs);
                    break; // Adapter ran to completion (shouldn't happen with real loops).
                }
                Err(GatewayError::Shutdown) => {
                    tracing::info!(platform = %platform, "adapter shutting down gracefully");
                    break;
                }
                Err(e) => {
                    tracing::warn!(
                        platform = %platform,
                        error = %e,
                        attempts = attempts,
                        "adapter disconnected, reconnecting"
                    );
                }
            }

            attempts += 1;
            if attempts >= config.max_attempts {
                tracing::error!(
                    platform = %platform,
                    max_attempts = config.max_attempts,
                    "max reconnect attempts reached, giving up"
                );
                break;
            }

            tracing::info!(
                platform = %platform,
                backoff_secs = backoff.as_secs(),
                "reconnecting in"
            );
            tokio::time::sleep(backoff).await;
            // Exponential backoff with deterministic jitter.
            let seed = (attempts as u64).wrapping_mul(0x9E3779B97F4A7C15);
            backoff = config.compute_backoff(backoff, seed).min(max_backoff);
        }
    }

    /// Route an inbound message through the agent.
    async fn route_inbound(&self, msg: InboundMessage) -> Result<(), GatewayError> {
        let user = PlatformUser {
            platform: msg.platform.clone(),
            user_id: msg.user_id.clone(),
        };

        // Check authorization.
        let adapter = self
            .adapters
            .iter()
            .find(|a| a.platform_name() == msg.platform);

        if let Some(adapter) = adapter {
            if !adapter.is_authorized(&msg.user_id) {
                tracing::warn!(
                    user = %user,
                    "unauthorized user rejected"
                );
                return Err(GatewayError::Auth(format!(
                    "user {} not authorized on {}",
                    msg.user_id, msg.platform
                )));
            }
        }

        // Update conversation state.
        {
            let mut convs = self.conversations.write().await;
            let state = convs.entry(user.clone()).or_insert_with(|| {
                tracing::info!(
                    user = %user,
                    conversation_id = "new",
                    "new conversation"
                );
                ConversationState::new(user.clone())
            });
            state.touch();
        }

        // Handle slash commands.
        if let Some((cmd, rest)) = parse_slash_command(&msg.content) {
            let response = self.handle_slash_command(&cmd, &rest).await;
            if let Err(e) = self.route_to_user(&msg.platform.to_string(), &msg.user_id, &response) {
                tracing::error!(error = %e, "failed to send slash command response");
            }
            return Ok(());
        }

        // Forward to agent core.
        let source = format!("{}:{}", msg.platform, msg.user_id);
        let agent_input = AgentInput::Gateway {
            source: source.clone(),
            content: msg.content.clone(),
        };

        if let Err(e) = self.agent_input_tx.send(agent_input) {
            return Err(GatewayError::Send(format!(
                "agent input channel error: {e}"
            )));
        }

        tracing::info!(
            source = %source,
            platform = %msg.platform,
            user = %msg.user_id,
            "message forwarded to agent"
        );

        Ok(())
    }

    /// Route agent output back to the originating platform and user.
    async fn route_outbound(&self, output: AgentOutput) -> Result<(), GatewayError> {
        let convs = self.conversations.read().await;

        if convs.is_empty() {
            tracing::warn!("no active conversations to route response to");
            return Ok(());
        }

        // Find the most recently active conversation and route to it.
        // In production, AgentOutput would carry a routing key (source platform + user_id).
        let mut latest_user: Option<&PlatformUser> = None;
        let mut latest_time: chrono::DateTime<chrono::Utc> = chrono::DateTime::from_timestamp_secs(0).unwrap_or_default();

        for (user, state) in convs.iter() {
            if state.last_active > latest_time {
                latest_time = state.last_active;
                latest_user = Some(user);
            }
        }

        if let Some(user) = latest_user {
            self.route_to_user(&user.platform.to_string(), &user.user_id, &output.content)?;
            tracing::info!(
                user = %user,
                platform = %user.platform,
                "response delivered"
            );
        }

        Ok(())
    }

    /// Route a message to a specific user on a specific platform.
    ///
    /// # Arguments
    /// * `source` - Platform name (e.g., "telegram", "discord").
    /// * `user_id` - The user identifier on that platform.
    /// * `content` - Message content to deliver.
    pub fn route_to_user(
        &self,
        source: &str,
        user_id: &str,
        content: &str,
    ) -> Result<(), GatewayError> {
        // Parse the platform from the source string.
        let platform = match source.to_lowercase().as_str() {
            "telegram" => Platform::Telegram,
            "discord" => Platform::Discord,
            "slack" => Platform::Slack,
            "whatsapp" => Platform::WhatsApp,
            "signal" => Platform::Signal,
            "email" => Platform::Email,
            _ => {
                return Err(GatewayError::Connection(format!(
                    "unknown platform in route_to_user: {}", source
                )));
            }
        };

        let adapter = self
            .adapters
            .iter()
            .find(|a| a.platform_name() == platform)
            .ok_or_else(|| {
                GatewayError::Connection(format!(
                    "no active adapter for platform {}", source
                ))
            })?;

        // We can't call async send_message from a sync method, so we store the
        // pending delivery and process it via the event loop. For synchronous use
        // (e.g., slash command replies), we spawn a one-shot task.
        let adapter = adapter.clone();
        let user_id = user_id.to_string();
        let content = content.to_string();

        tokio::spawn(async move {
            if let Err(e) = adapter.send_message(&user_id, &content).await {
                tracing::error!(
                    platform = %platform,
                    user = %user_id,
                    error = %e,
                    "route_to_user delivery failed"
                );
            } else {
                tracing::debug!(
                    platform = %platform,
                    user = %user_id,
                    "route_to_user delivered"
                );
            }
        });

        Ok(())
    }

    /// Send a reply to a specific user on a platform.
    async fn reply_to_user(
        &self,
        platform: &Platform,
        user_id: &str,
        content: &str,
    ) -> Result<(), GatewayError> {
        let adapter = self
            .adapters
            .iter()
            .find(|a| a.platform_name() == *platform);

        match adapter {
            Some(adapter) => adapter.send_message(user_id, content).await,
            None => Err(GatewayError::Connection(format!(
                "no adapter for platform {}", platform
            ))),
        }
    }

    /// Handle a slash command and return a text response.
    async fn handle_slash_command(&self, cmd: &SlashCommand, _args: &str) -> String {
        match cmd {
            SlashCommand::Help => {
                "Available commands:\n  /help     - Show this message\n  /reset  - Reset conversation\n  /status - Show gateway status\n  /model  - Show current model\n  /tools  - List available tools".to_string()
            }
            SlashCommand::Reset => {
                tracing::info!("slash command: reset");
                "Conversation context cleared.".to_string()
            }
            SlashCommand::Status => {
                let adapter_count = self.adapters.len();
                let conv_count = self.conversations.read().await.len();
                format!(
                    "Gateway status:\n  Active adapters: {adapter_count}\n  Conversations: {conv_count}"
                )
            }
            SlashCommand::Model => {
                "Model information available via the agent core.".to_string()
            }
            SlashCommand::Tools => {
                "Tool list available via the agent core.".to_string()
            }
            SlashCommand::Custom(name, _args) => {
                format!("Unknown command: /{name}. Type /help for available commands.")
            }
        }
    }

    /// Signal the gateway to shut down.
    pub fn shutdown(&self) {
        self.shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the current number of active platform adapters.
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// Get the current number of tracked conversations.
    pub async fn conversation_count(&self) -> usize {
        self.conversations.read().await.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_display() {
        assert_eq!(format!("{}", Platform::Telegram), "telegram");
        assert_eq!(format!("{}", Platform::Discord), "discord");
        assert_eq!(format!("{}", Platform::Slack), "slack");
        assert_eq!(format!("{}", Platform::WhatsApp), "whatsapp");
        assert_eq!(format!("{}", Platform::Signal), "signal");
        assert_eq!(format!("{}", Platform::Email), "email");
    }

    #[test]
    fn platform_user_display() {
        let user = PlatformUser {
            platform: Platform::Telegram,
            user_id: "12345".into(),
        };
        assert_eq!(format!("{user}"), "telegram:12345");
    }

    #[test]
    fn parse_slash_command_help() {
        let (cmd, rest) = parse_slash_command("/help").unwrap();
        assert!(matches!(cmd, SlashCommand::Help));
        assert!(rest.is_empty());
    }

    #[test]
    fn parse_slash_command_with_args() {
        let (cmd, rest) = parse_slash_command("/status check").unwrap();
        assert!(matches!(cmd, SlashCommand::Status));
        assert_eq!(rest, "check");
    }

    #[test]
    fn parse_slash_command_custom() {
        let (cmd, rest) = parse_slash_command("/custom arg1 arg2").unwrap();
        match cmd {
            SlashCommand::Custom(name, _) => assert_eq!(name, "custom"),
            _ => panic!("expected custom command"),
        }
        assert_eq!(rest, "arg1 arg2");
    }

    #[test]
    fn parse_slash_command_no_slash() {
        assert!(parse_slash_command("hello world").is_none());
    }

    #[test]
    fn parse_slash_command_case_insensitive() {
        let (cmd, _) = parse_slash_command("/HELP").unwrap();
        assert!(matches!(cmd, SlashCommand::Help));
    }

    #[test]
    fn conversation_state_tracks_messages() {
        let user = PlatformUser {
            platform: Platform::Discord,
            user_id: "abc".into(),
        };
        let mut state = ConversationState::new(user);
        assert_eq!(state.message_count, 0);
        state.touch();
        assert_eq!(state.message_count, 1);
        state.touch();
        assert_eq!(state.message_count, 2);
    }

    #[test]
    fn slash_command_names() {
        assert_eq!(SlashCommand::Help.name(), "help");
        assert_eq!(SlashCommand::Reset.name(), "reset");
        assert_eq!(SlashCommand::Status.name(), "status");
        assert_eq!(SlashCommand::Model.name(), "model");
        assert_eq!(SlashCommand::Tools.name(), "tools");
        assert_eq!(
            SlashCommand::Custom("foo".into(), vec![]).name(),
            "foo"
        );
    }

    #[test]
    fn gateway_config_default() {
        let cfg = GatewayConfig::default();
        assert!(cfg.telegram.is_none());
        assert!(cfg.discord.is_none());
        assert_eq!(cfg.poll_interval_secs, 5);
    }

    #[test]
    fn gateway_new_with_empty_config() {
        let (input_tx, _) = mpsc::unbounded_channel::<AgentInput>();
        let (_, output_rx) = mpsc::unbounded_channel::<AgentOutput>();

        let gw = Gateway::new(GatewayConfig::default(), input_tx, output_rx);
        assert_eq!(gw.adapter_count(), 0);
    }

    #[test]
    fn gateway_new_with_telegram() {
        let (input_tx, _) = mpsc::unbounded_channel::<AgentInput>();
        let (_, output_rx) = mpsc::unbounded_channel::<AgentOutput>();

        let cfg = GatewayConfig {
            telegram: Some(TelegramConfig {
                bot_token: "test".into(),
                allowed_user_ids: vec![],
            }),
            ..Default::default()
        };

        let gw = Gateway::new(cfg, input_tx, output_rx);
        assert_eq!(gw.adapter_count(), 1);
    }

    #[test]
    fn gateway_new_with_all_platforms() {
        let (input_tx, _) = mpsc::unbounded_channel::<AgentInput>();
        let (_, output_rx) = mpsc::unbounded_channel::<AgentOutput>();

        let cfg = GatewayConfig {
            telegram: Some(TelegramConfig {
                bot_token: "t".into(),
                allowed_user_ids: vec![],
            }),
            discord: Some(DiscordConfig {
                bot_token: "d".into(),
                application_id: "1".into(),
                allowed_user_ids: vec![],
            }),
            slack: Some(SlackConfig {
                bot_token: "s".into(),
                app_token: "a".into(),
                allowed_user_ids: vec![],
            }),
            whatsapp: Some(WhatsAppConfig {
                api_key: "w".into(),
                phone_number_id: "p".into(),
                allowed_user_ids: vec![],
            }),
            signal: Some(SignalConfig {
                recipient_number: "r".into(),
                gateway_url: "g".into(),
                allowed_user_ids: vec![],
            }),
            email: Some(EmailConfig {
                imap_host: "i".into(),
                imap_port: 993,
                imap_user: "u".into(),
                imap_password: "p".into(),
                smtp_host: "s".into(),
                smtp_port: 587,
                smtp_user: "u".into(),
                smtp_password: "p".into(),
                allowed_senders: vec![],
            }),
            poll_interval_secs: 5,
        };

        let gw = Gateway::new(cfg, input_tx, output_rx);
        assert_eq!(gw.adapter_count(), 6);
    }

    #[tokio::test]
    async fn conversation_count_starts_at_zero() {
        let (input_tx, _) = mpsc::unbounded_channel::<AgentInput>();
        let (_, output_rx) = mpsc::unbounded_channel::<AgentOutput>();

        let gw = Gateway::new(GatewayConfig::default(), input_tx, output_rx);
        assert_eq!(gw.conversation_count().await, 0);
    }

    #[test]
    fn telegram_adapter_authorization_empty_allowlist() {
        let adapter = TelegramAdapter::new(TelegramConfig {
            bot_token: "test".into(),
            allowed_user_ids: vec![],
        });
        // Empty allowlist means everyone is authorized.
        assert!(adapter.is_authorized("anyone"));
    }

    #[test]
    fn telegram_adapter_authorization_with_allowlist() {
        let adapter = TelegramAdapter::new(TelegramConfig {
            bot_token: "test".into(),
            allowed_user_ids: vec!["123".into(), "456".into()],
        });
        assert!(adapter.is_authorized("123"));
        assert!(adapter.is_authorized("456"));
        assert!(!adapter.is_authorized("999"));
    }

    #[test]
    fn email_adapter_authorization_with_allowlist() {
        let adapter = EmailAdapter::new(EmailConfig {
            imap_host: "mail.example.com".into(),
            imap_port: 993,
            imap_user: "bot@example.com".into(),
            imap_password: "secret".into(),
            smtp_host: "smtp.example.com".into(),
            smtp_port: 587,
            smtp_user: "bot@example.com".into(),
            smtp_password: "secret".into(),
            allowed_senders: vec!["alice@example.com".into()],
        });
        assert!(adapter.is_authorized("alice@example.com"));
        assert!(!adapter.is_authorized("bob@example.com"));
    }

    #[test]
    fn route_to_user_unknown_platform() {
        let (input_tx, _) = mpsc::unbounded_channel::<AgentInput>();
        let (_, output_rx) = mpsc::unbounded_channel::<AgentOutput>();
        let gw = Gateway::new(GatewayConfig::default(), input_tx, output_rx);

        let result = gw.route_to_user("matrix", "user123", "hello");
        assert!(result.is_err());
    }

    #[test]
    fn route_to_user_no_adapter() {
        let (input_tx, _) = mpsc::unbounded_channel::<AgentInput>();
        let (_, output_rx) = mpsc::unbounded_channel::<AgentOutput>();
        let gw = Gateway::new(GatewayConfig::default(), input_tx, output_rx);

        // Telegram is not configured, so no adapter exists.
        let result = gw.route_to_user("telegram", "user123", "hello");
        assert!(result.is_err());
    }

    #[test]
    fn route_to_user_with_adapter() {
        let (input_tx, _) = mpsc::unbounded_channel::<AgentInput>();
        let (_, output_rx) = mpsc::unbounded_channel::<AgentOutput>();

        let cfg = GatewayConfig {
            telegram: Some(TelegramConfig {
                bot_token: "test".into(),
                allowed_user_ids: vec![],
            }),
            ..Default::default()
        };

        let gw = Gateway::new(cfg, input_tx, output_rx);

        // Should succeed (spawns a task that will fail on actual API call).
        let result = gw.route_to_user("telegram", "12345", "hello");
        assert!(result.is_ok());
    }
}
