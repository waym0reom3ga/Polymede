use std::sync::Arc;

use clap::{Parser, Subcommand};
use std::io::Write;
use tokio::signal;

use polymede::config::Config;
use polymede::tui::TuiApp;

#[derive(Parser)]
#[command(name = "polymede")]
#[command(about = "Polymede AI agent with persistent memory and tool use")]
#[command(version)]
struct PolymedeCli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Configure providers, API keys, and options
    Setup,
    /// Choose LLM provider and model
    Model,
    /// Enable or disable tools
    Tools,
    /// Gateway subcommands
    #[command(subcommand)]
    Gateway(GatewayCommands),
    /// Set individual config values
    Config {
        /// Action to perform
        #[arg(long = "set")]
        set: bool,
        /// Config key (e.g. llm.provider, tools.enabled)
        key: Option<String>,
        /// Config value
        value: Option<String>,
    },
    /// Diagnose configuration and environment issues
    Doctor,
    /// Update to the latest version
    Update,
}

#[derive(Subcommand)]
enum GatewayCommands {
    /// Start the messaging gateway
    Start,
    /// Configure messaging platforms
    Setup,
}

#[tokio::main]
async fn main() {
    let cli = PolymedeCli::parse();

    let config = load_config_or_default().await;
    let log_level = config.as_ref().map(|c| c.logging.level.clone()).unwrap_or_default();
    init_logging(&log_level);

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        "polymede starting"
    );

    let shutdown_rx = make_shutdown_rx();

    match cli.command {
        None => {
            // Default: launch interactive TUI
            if let Err(e) = run_tui(config, shutdown_rx).await {
                tracing::error!(error = %e, "polymede exited with error");
            }
        }
        Some(Commands::Setup) => {
            // Interactive setup wizard
            if let Err(e) = run_setup_wizard(config, shutdown_rx).await {
                tracing::error!(error = %e, "setup failed");
            }
        }
        Some(Commands::Model) => {
            if let Err(e) = run_model(config, shutdown_rx).await {
                tracing::error!(error = %e, "model command failed");
            }
        }
        Some(Commands::Tools) => {
            if let Err(e) = run_tools(config, shutdown_rx).await {
                tracing::error!(error = %e, "tools command failed");
            }
        }
        Some(Commands::Gateway(gw)) => {
            if let Err(e) = run_gateway(gw, config, shutdown_rx).await {
                tracing::error!(error = %e, "gateway command failed");
            }
        }
        Some(Commands::Config { set, key, value }) => {
            if let Err(e) = run_config(set, key, value, config, shutdown_rx).await {
                tracing::error!(error = %e, "config command failed");
            }
        }
        Some(Commands::Doctor) => {
            if let Err(e) = run_doctor(config, shutdown_rx).await {
                tracing::error!(error = %e, "doctor command failed");
            }
        }
        Some(Commands::Update) => {
            if let Err(e) = run_update(config, shutdown_rx).await {
                tracing::error!(error = %e, "update command failed");
            }
        }
    }
}

async fn load_config_or_default() -> Option<Config> {
    match Config::load() {
        Ok(c) => Some(c),
        Err(_) => {
            tracing::warn!("config load failed, using defaults");
            Config::create_default().ok()
        }
    }
}

fn init_logging(level: &str) {
    let default_level = if level.is_empty() { "info" } else { level };
    let _ = tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(parse_tracing_level(default_level))
        .try_init();
}

fn parse_tracing_level(level: &str) -> tracing::level_filters::LevelFilter {
    match level.to_lowercase().as_str() {
        "trace" => tracing::level_filters::LevelFilter::TRACE,
        "debug" => tracing::level_filters::LevelFilter::DEBUG,
        "warn" => tracing::level_filters::LevelFilter::WARN,
        "error" => tracing::level_filters::LevelFilter::ERROR,
        _ => tracing::level_filters::LevelFilter::INFO,
    }
}

fn make_shutdown_rx() -> Arc<tokio::sync::mpsc::Receiver<()>> {
    let (_, rx) = tokio::sync::mpsc::channel(1);
    Arc::new(rx)
}

async fn run_tui(
    config: Option<Config>,
    _shutdown_rx: Arc<tokio::sync::mpsc::Receiver<()>>,
) -> Result<(), String> {
    let config = config.ok_or_else(|| "no config available".to_string())?;

    tracing::info!("starting interactive TUI");

    let app = TuiApp::new(config).await?;

    let run_task = app.run();

    tokio::select! {
        result = run_task => result,
        _ = wait_shutdown() => {
            tracing::info!("shutdown signal received");
            Ok(())
        }
    }
}

/// Interactive setup wizard: asks for provider, model, API key, base URL.
async fn run_setup_wizard(
    config: Option<Config>,
    _shutdown_rx: Arc<tokio::sync::mpsc::Receiver<()>>,
) -> Result<(), String> {
    let mut cfg = match config {
        Some(c) => c,
        None => Config::create_default().map_err(|e| format!("cannot create default config: {e}"))?,
    };

    println!("=== Polymede Setup ===\n");

    // Provider
    println!("Supported providers: openrouter, lmstudio, ollama, anthropic, custom");
    print!("Provider [{}]: ", cfg.llm.provider);
    std::io::stdout().flush().ok();
    let line = read_line()?;
    if !line.is_empty() {
        cfg.llm.provider = line.trim().to_string();
    }

    // Model
    print!("Model [{}]: ", cfg.llm.model);
    std::io::stdout().flush().ok();
    let line = read_line()?;
    if !line.is_empty() {
        cfg.llm.model = line.trim().to_string();
    }

    // API key
    let current_key_display = match &cfg.llm.api_key {
        Some(k) if k.len() > 8 => format!("{}****", &k[..4]),
        Some(_) => "set".into(),
        None => "not set".into(),
    };
    print!("API key (leave empty to skip, current: {}): ", current_key_display);
    std::io::stdout().flush().ok();
    let line = read_line()?;
    if !line.is_empty() {
        cfg.llm.api_key = Some(line.trim().to_string());
    }

    // Base URL (optional)
    let current_url = cfg.llm.base_url.as_deref().unwrap_or("(auto)");
    print!("Base URL [{}]: ", current_url);
    std::io::stdout().flush().ok();
    let line = read_line()?;
    if !line.is_empty() {
        cfg.llm.base_url = Some(line.trim().to_string());
    }

    // Save
    cfg.save().map_err(|e| format!("failed to save config: {e}"))?;

    println!("\nConfig saved to {:?}", Config::config_path());
    println!("Run 'polymede' to start the agent.");

    Ok(())
}

/// Helper: read a line from stdin.
fn read_line() -> Result<String, String> {
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|e| format!("failed to read input: {e}"))?;
    Ok(input)
}

async fn run_model(
    config: Option<Config>,
    _shutdown_rx: Arc<tokio::sync::mpsc::Receiver<()>>,
) -> Result<(), String> {
    let config = config.ok_or_else(|| "no config available".to_string())?;

    tracing::info!("model selection command");
    println!("Current provider: {}", config.llm.provider);
    println!("Current model: {}", config.llm.model);
    println!("Use 'polymede setup' to change model configuration.");
    Ok(())
}

async fn run_tools(
    config: Option<Config>,
    _shutdown_rx: Arc<tokio::sync::mpsc::Receiver<()>>,
) -> Result<(), String> {
    let config = config.ok_or_else(|| "no config available".to_string())?;

    tracing::info!("tools management command");
    println!("Enabled tools:");
    for tool in &config.tools.enabled {
        println!("  - {tool}");
    }
    println!("\nUse 'polymede setup' to enable or disable tools.");
    Ok(())
}

async fn run_gateway(
    gw: GatewayCommands,
    _config: Option<Config>,
    _shutdown_rx: Arc<tokio::sync::mpsc::Receiver<()>>,
) -> Result<(), String> {
    match gw {
        GatewayCommands::Start => {
            tracing::info!("gateway start command");
            println!("Messaging gateway not yet configured.");
            println!("Run 'polymede gateway setup' first.");
            Ok(())
        }
        GatewayCommands::Setup => {
            tracing::info!("gateway setup command");
            println!("Gateway setup not yet implemented.");
            Ok(())
        }
    }
}

async fn run_config(
    set: bool,
    key: Option<String>,
    value: Option<String>,
    config: Option<Config>,
    _shutdown_rx: Arc<tokio::sync::mpsc::Receiver<()>>,
) -> Result<(), String> {
    if !set {
        if let Some(cfg) = config {
            let serialized = toml::to_string_pretty(&cfg).map_err(|e| e.to_string())?;
            println!("Current config:");
            println!("{serialized}");
        } else {
            println!("No config loaded.");
        }
        return Ok(());
    }

    let (key, value) = match (key, value) {
        (Some(k), Some(v)) => (k, v),
        _ => {
            eprintln!("Usage: polymede config --set <key> <value>");
            return Ok(());
        }
    };

    tracing::info!(key = %key, "setting config value");
    println!("Config key '{key}' = '{value}'");
    println!("Use 'polymede setup' for interactive configuration.");
    Ok(())
}

async fn run_doctor(
    config: Option<Config>,
    _shutdown_rx: Arc<tokio::sync::mpsc::Receiver<()>>,
) -> Result<(), String> {
    let config = match config {
        Some(c) => c,
        None => {
            println!("No config loaded. Run 'polymede setup' first.");
            return Ok(());
        }
    };

    tracing::info!("running diagnostics");
    println!("Polymede diagnostics");
    println!("==================");

    let mut issues: Vec<String> = Vec::new();

    match config.validate() {
        Ok(()) => println!("[OK] Config validation passed"),
        Err(errors) => {
            println!("[FAIL] Config validation failed:");
            for e in &errors {
                println!("  - {e}");
                issues.push(e.clone());
            }
        }
    }

    let state_dir = Config::state_dir();
    if state_dir.exists() {
        println!("[OK] State directory exists: {:?}", state_dir);
    } else {
        println!("[WARN] State directory not found: {:?}", state_dir);
    }

    let db_path = config.db_path();
    if db_path.exists() {
        println!("[OK] Memory database exists: {:?}", db_path);
    } else {
        println!("[INFO] Memory database will be created on first use");
    }

    let skill_dir = config.skill_dir();
    if skill_dir.exists() {
        println!("[OK] Skills directory exists: {:?}", skill_dir);
    } else {
        println!("[INFO] Skills directory not found: {:?}", skill_dir);
    }

    if let Some(key) = config.effective_api_key() {
        let masked = format!("{}****", &key[..key.len().min(4)]);
        println!("[OK] API key configured: {masked}");
    } else {
        println!("[FAIL] No API key configured");
        issues.push("No API key set".into());
    }

    println!("\nProvider: {}", config.llm.provider);
    println!("Model:    {}", config.llm.model);
    println!("Tools:    {}", config.tools.enabled.join(", "));

    if issues.is_empty() {
        println!("\nAll checks passed.");
    } else {
        println!("\n{} issue(s) found.", issues.len());
    }

    Ok(())
}

async fn run_update(
    _config: Option<Config>,
    _shutdown_rx: Arc<tokio::sync::mpsc::Receiver<()>>,
) -> Result<(), String> {
    tracing::info!("update command");
    println!("Current version: {}", env!("CARGO_PKG_VERSION"));
    println!("Use 'cargo install --force polymede' to update.");
    Ok(())
}

async fn wait_shutdown() {
    tokio::select! {
        _ = signal::ctrl_c() => {
            tracing::info!("received Ctrl+C");
        }
        _ = signal_unix_shutdown() => {
            tracing::info!("received shutdown signal");
        }
    }
}

#[cfg(unix)]
async fn signal_unix_shutdown() {
    let mut term = signal::unix::signal(signal::unix::SignalKind::terminate())
        .expect("create SIGTERM handler");
    let _ = term.recv().await;
}

#[cfg(not(unix))]
async fn signal_unix_shutdown() {
    let () = tokio::signal::ctrl_c().await;
}
