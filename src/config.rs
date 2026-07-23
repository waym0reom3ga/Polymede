use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const CONFIG_DIR: &str = "polymede";
const CONFIG_FILE: &str = "config.toml";
const STATE_DIR: &str = "polymede";
const ENV_API_KEY: &str = "POLYMDE_LLM_API_KEY";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<Box<LlmConfig>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    #[serde(default = "default_tools")]
    pub enabled: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    #[serde(default = "default_compression_interval")]
    pub compression_interval: String,
    #[serde(default = "default_max_recall_tokens")]
    pub max_recall_tokens: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
}

fn default_tools() -> Vec<String> {
    vec!["terminal".into(), "file".into()]
}

fn default_compression_interval() -> String {
    "6h".into()
}

fn default_max_recall_tokens() -> usize {
    200_000
}

fn default_log_level() -> String {
    "info".into()
}

impl Default for ToolsConfig {
    fn default() -> Self {
        ToolsConfig {
            enabled: default_tools(),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            compression_interval: default_compression_interval(),
            max_recall_tokens: default_max_recall_tokens(),
        }
    }
}

impl Config {
    /// Load config from file, apply env overrides, create default if missing.
    pub fn load() -> Result<Self, ConfigError> {
        let path = Self::config_path();

        let mut config: Config = if path.exists() {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| ConfigError::Read(format!("cannot read {path:?}: {e}")))?;
            toml::from_str(&content)
                .map_err(|e| ConfigError::Parse(format!("invalid TOML in {path:?}: {e}")))?
        } else {
            tracing::info!("no config found, creating default");
            return Self::create_default();
        };

        config.apply_env_overrides();
        Ok(config)
    }

    /// Write the current config back to disk.
    pub fn save(&self) -> Result<(), ConfigError> {
        let path = Self::config_path();
        let parent = path.parent().ok_or(ConfigError::NoConfigDir)?;
        std::fs::create_dir_all(parent)
            .map_err(|e| ConfigError::Io(format!("cannot create config dir: {e}")))?;

        let content = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::Serialize(format!("cannot serialize config: {e}")))?;
        std::fs::write(&path, content)
            .map_err(|e| ConfigError::Write(format!("cannot write {path:?}: {e}")))?;

        tracing::info!("config saved to {path:?}");
        Ok(())
    }

    /// Create a default config file and return it.
    pub fn create_default() -> Result<Self, ConfigError> {
        let path = Self::config_path();
        let parent = path.parent().ok_or(ConfigError::NoConfigDir)?;
        std::fs::create_dir_all(parent)
            .map_err(|e| ConfigError::Io(format!("cannot create config dir: {e}")))?;

        let config = Self::default_config();
        let content = toml::to_string_pretty(&config)
            .map_err(|e| ConfigError::Serialize(format!("cannot serialize default: {e}")))?;
        std::fs::write(&path, content)
            .map_err(|e| ConfigError::Write(format!("cannot write default: {e}")))?;

        tracing::info!("default config created at {path:?}");
        Ok(config)
    }

    /// Path to the SQLite memory database.
    pub fn db_path(&self) -> PathBuf {
        Self::state_dir().join("memory.db")
    }

    /// Path to the skills directory.
    pub fn skill_dir(&self) -> PathBuf {
        Self::state_dir().join("skills")
    }

    /// Effective API key (env var takes precedence over file).
    pub fn effective_api_key(&self) -> Option<String> {
        std::env::var(ENV_API_KEY).ok().or_else(|| self.llm.api_key.clone())
    }

    /// Check that the config has the minimum required fields.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.llm.provider.is_empty() {
            errors.push("llm.provider is required".into());
        }
        if self.llm.model.is_empty() {
            errors.push("llm.model is required".into());
        }
        if self.effective_api_key().is_none() {
            errors.push(
                "llm.api_key is required (set in config or via POLYMDE_LLM_API_KEY env var)"
                    .into(),
            );
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(CONFIG_DIR)
            .join(CONFIG_FILE)
    }

    pub fn state_dir() -> PathBuf {
        dirs::state_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(STATE_DIR)
    }

    fn default_config() -> Self {
        Config {
            llm: LlmConfig {
                provider: "openrouter".into(),
                model: "anthropic/claude-sonnet-4-20250514".into(),
                api_key: None,
                base_url: None,
                fallback: Some(Box::new(LlmConfig {
                    provider: "lmstudio".into(),
                    model: "qwen3-27b".into(),
                    api_key: None,
                    base_url: Some("http://localhost:1234/v1".into()),
                    fallback: None,
                })),
            },
            tools: ToolsConfig {
                enabled: vec![
                    "terminal".into(),
                    "file".into(),
                    "web_search".into(),
                    "mcp".into(),
                ],
            },
            memory: MemoryConfig {
                compression_interval: "6h".into(),
                max_recall_tokens: 200_000,
            },
            logging: LoggingConfig {
                level: "info".into(),
            },
        }
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(key) = std::env::var(ENV_API_KEY) {
            tracing::debug!("overriding LLM API key from environment");
            self.llm.api_key = Some(key);
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Read(String),
    Write(String),
    Parse(String),
    Serialize(String),
    Io(String),
    NoConfigDir,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Read(msg) => write!(f, "config read error: {msg}"),
            ConfigError::Write(msg) => write!(f, "config write error: {msg}"),
            ConfigError::Parse(msg) => write!(f, "config parse error: {msg}"),
            ConfigError::Serialize(msg) => write!(f, "config serialize error: {msg}"),
            ConfigError::Io(msg) => write!(f, "config IO error: {msg}"),
            ConfigError::NoConfigDir => write!(f, "cannot determine config directory"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let mut config = Config::default_config();
        config.llm.api_key = Some("test-key".into());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn missing_provider_fails_validation() {
        let mut config = Config::default_config();
        config.llm.provider.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn env_api_key_overrides_none() {
        let config = Config::default_config();
        unsafe { std::env::set_var(ENV_API_KEY, "env-key") };
        assert_eq!(config.effective_api_key(), Some("env-key".into()));
        unsafe { std::env::remove_var(ENV_API_KEY) };
    }

    #[test]
    fn env_api_key_overrides_file() {
        let mut config = Config::default_config();
        config.llm.api_key = Some("file-key".into());
        unsafe { std::env::set_var(ENV_API_KEY, "env-key") };
        assert_eq!(config.effective_api_key(), Some("env-key".into()));
        unsafe { std::env::remove_var(ENV_API_KEY) };
    }

    #[test]
    fn db_path_contains_memory_db() {
        let config = Config::default_config();
        assert!(config.db_path().ends_with("memory.db"));
    }

    #[test]
    fn skill_dir_ends_with_skills() {
        let config = Config::default_config();
        assert!(config.skill_dir().ends_with("skills"));
    }

    #[test]
    fn roundtrip_serialize_deserialize() {
        let config = Config::default_config();
        let serialized = toml::to_string_pretty(&config).expect("serialize");
        let deserialized: Config = toml::from_str(&serialized).expect("deserialize");
        assert_eq!(config.llm.provider, deserialized.llm.provider);
        assert_eq!(config.llm.model, deserialized.llm.model);
    }
}
