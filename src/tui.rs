use std::io;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use ratatui::Terminal;
use rustyline::config::Configurer;

use rustyline::history::DefaultHistory;
use rustyline::{CompletionType, Config as RlConfig, Editor};
use tokio::sync::mpsc;
use std::sync::Mutex;

use crate::agent::{Agent, AgentInput, AgentOutput};
use crate::config::Config;
use crate::memory::MemoryIntegration;

// ---------------------------------------------------------------------------
// Slash commands registry
// ---------------------------------------------------------------------------

const SLASH_COMMANDS: &[&str] = &[
    "/help",
    "/exit",
    "/queue",
    "/provider",
    "/model",
    "/new",
    "/reset",
    "/skills",
    "/compress",
    "/usage",
    "/insights",
    "/stop",
];

// ---------------------------------------------------------------------------
// TuiCompleter - rustyline autocomplete for slash commands
// ---------------------------------------------------------------------------



// ---------------------------------------------------------------------------
// TUI state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum TuiMessage {
    UserInput(String),
    AssistantText(String),
    ToolOutput { name: String, output: String, ok: bool },
    System(String),
}

struct SharedState {
    current_response: String,
    is_processing: bool,
    tool_outputs: Vec<TuiMessage>,
    total_prompt_tokens: u32,
    total_completion_tokens: u32,
    history: Vec<TuiMessage>,
    model: String,
    provider: String,
    memory_status: String,
    interrupted: bool,
}

impl SharedState {
    fn new(model: String, provider: String) -> Self {
        Self {
            current_response: String::new(),
            is_processing: false,
            tool_outputs: Vec::new(),
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            history: Vec::new(),
            model,
            provider,
            memory_status: "ready".into(),
            interrupted: false,
        }
    }
}

// ---------------------------------------------------------------------------
// TuiApp
// ---------------------------------------------------------------------------

pub struct TuiApp {
    state: Arc<Mutex<SharedState>>,
    input_tx: mpsc::UnboundedSender<AgentInput>,
    agent_shutdown: mpsc::UnboundedSender<()>,
    editor: Editor<(), DefaultHistory>,
    current_model: String,
    config: Config,
    memory: MemoryIntegration,
}

impl TuiApp {
    pub async fn new(config: Config) -> Result<Self, String> {
        let state_dir = Config::state_dir();

        let agent_memory = MemoryIntegration::new(
            state_dir.clone(),
            config.memory.clone(),
            Some(config.llm.clone()),
            true,
        )
        .await
        .map_err(|e| format!("memory init failed: {e}"))?;

        let tui_memory = MemoryIntegration::new(
            state_dir.clone(),
            config.memory.clone(),
            Some(config.llm.clone()),
            false,
        )
        .await
        .map_err(|e| format!("memory init failed: {e}"))?;

        let (agent, input_tx, mut output_rx, mut chunk_rx) = Agent::new(config.clone(), agent_memory)
            .await
            .map_err(|e| format!("agent init failed: {e}"))?;

        let (shutdown_tx, mut shutdown_rx) = mpsc::unbounded_channel();

        let model = config.llm.model.clone();
        let provider = config.llm.provider.clone();
        let shared_state = Arc::new(Mutex::new(SharedState::new(model.clone(), provider)));

        let state_clone = Arc::clone(&shared_state);
        tokio::spawn(async move {
            let agent_task = agent.run();
            let output_loop = Self::output_loop(&mut output_rx, &state_clone);
            let chunk_loop = Self::chunk_loop(&mut chunk_rx, &state_clone);

            tokio::select! {
                biased;
                _ = shutdown_rx.recv() => {
                    tracing::info!("tui shutdown signal");
                }
                result = agent_task => {
                    match result {
                        Ok(()) => tracing::info!("agent exited cleanly"),
                        Err(e) => tracing::error!(error = %e, "agent error"),
                    }
                }
                _ = output_loop => {}
                _ = chunk_loop => {}
            }
        });

        let rl_config = RlConfig::builder()
            .max_history_size(500)
            .map_err(|e| format!("rustyline config failed: {e}"))?
            .completion_type(CompletionType::List)
            .build();

        let mut editor = Editor::<(), DefaultHistory>::with_config(rl_config)
            .map_err(|e| format!("rustyline init failed: {e}"))?;
        editor.set_keyseq_timeout(Some(200));

        Ok(Self {
            state: shared_state,
            input_tx,
            agent_shutdown: shutdown_tx,
            editor,
            current_model: model,
            config,
            memory: tui_memory,
        })
    }

    async fn output_loop(
        output_rx: &mut mpsc::UnboundedReceiver<AgentOutput>,
        state: &Arc<Mutex<SharedState>>,
    ) {
        loop {
            match output_rx.recv().await {
                Some(output) => {
                    let mut st = state.lock().expect("mutex poisoned");
                    st.total_prompt_tokens += output.token_usage.prompt_tokens;
                    st.total_completion_tokens += output.token_usage.completion_tokens;

                    for tool in &output.tool_results {
                        st.tool_outputs.push(TuiMessage::ToolOutput {
                            name: tool.name.clone(),
                            output: tool.output.clone(),
                            ok: tool.ok,
                        });
                    }

                    st.history.push(TuiMessage::AssistantText(output.content.clone()));
                    st.current_response = String::new();
                    st.is_processing = false;
                    st.interrupted = false;
                }
                None => {
                    tracing::info!("output channel closed");
                    break;
                }
            }
        }
    }

    async fn chunk_loop(
        chunk_rx: &mut mpsc::UnboundedReceiver<String>,
        state: &Arc<Mutex<SharedState>>,
    ) {
        loop {
            match chunk_rx.recv().await {
                Some(chunk) => {
                    let mut st = state.lock().expect("mutex poisoned");
                    st.current_response.push_str(&chunk);
                }
                None => break,
            }
        }
    }

    async fn handle_slash_command(&mut self, cmd: &str) -> Option<TuiMessage> {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        let name = parts[0].to_lowercase();
        let arg = if parts.len() >= 2 && !parts[1].trim().is_empty() {
            Some(parts[1].trim())
        } else {
            None
        };

        match name.as_str() {
            "/help" => {
                let cmds: Vec<&str> = SLASH_COMMANDS.iter().map(|&s| s).collect();
                Some(TuiMessage::System(format!(
                    "Commands: {}",
                    cmds.join(", ")
                )))
            }

            "/exit" => {
                tracing::info!("exit requested via /exit");
                self.agent_shutdown.send(()).ok();
                Some(TuiMessage::System("Exiting...".into()))
            }

            "/queue" => {
                let st = self.state.lock().expect("mutex poisoned");
                let status = if st.is_processing {
                    "Processing a turn..."
                } else {
                    "Idle - no pending turns"
                };
                Some(TuiMessage::System(format!(
                    "Queue: {} | Messages in context: {}",
                    status,
                    st.history.len()
                )))
            }

            "/provider" => {
                let st = self.state.lock().expect("mutex poisoned");
                match arg {
                    Some(p) if !p.is_empty() => {
                        drop(st);
                        // Note: can't mutate self.config here due to borrow rules.
                        // SetProvider is sent to agent which updates its own config copy.
                        let _ = self.input_tx.send(AgentInput::SetProvider(p.to_string()));
                        Some(TuiMessage::System(format!("Switching provider to '{}'.", p)))
                    }
                    _ => {
                        Some(TuiMessage::System(format!(
                            "Current provider: {}",
                            st.provider
                        )))
                    }
                }
            }

            "/model" => {
                let st = self.state.lock().expect("mutex poisoned");
                match arg {
                    Some(m) if !m.is_empty() => {
                        drop(st);
                        self.current_model = m.to_string();
                        let _ = self.input_tx.send(AgentInput::SetModel(m.to_string()));
                        Some(TuiMessage::System(format!("Switching model to '{}'.", m)))
                    }
                    _ => {
                        Some(TuiMessage::System(format!(
                            "Current model: {} (provider: {})",
                            st.model, st.provider
                        )))
                    }
                }
            }

            "/new" | "/reset" => {
                tracing::info!("resetting conversation");
                let mut st = self.state.lock().expect("mutex poisoned");
                st.history.clear();
                st.current_response.clear();
                st.tool_outputs.clear();
                let _ = self.input_tx.send(AgentInput::Reset);
                Some(TuiMessage::System("Conversation reset.".into()))
            }

            "/stop" => {
                tracing::info!("interrupting current turn");
                let mut st = self.state.lock().expect("mutex poisoned");
                st.interrupted = true;
                st.is_processing = false;
                Some(TuiMessage::System("Turn interrupted.".into()))
            }

            "/usage" => {
                let st = self.state.lock().expect("mutex poisoned");
                let msg = format!(
                    "Token usage -- prompt: {}, completion: {}, total: {}",
                    st.total_prompt_tokens,
                    st.total_completion_tokens,
                    st.total_prompt_tokens + st.total_completion_tokens,
                );
                Some(TuiMessage::System(msg))
            }

            "/skills" => {
                let skill_dir = self.config.skill_dir();
                if skill_dir.exists() {
                    let mut skills = Vec::new();
                    if let Ok(mut entries) = tokio::fs::read_dir(&skill_dir).await {
                        while let Ok(Some(entry)) = entries.next_entry().await {
                            let name = entry.file_name();
                            skills.push(name.to_string_lossy().to_string());
                        }
                    }
                    if skills.is_empty() {
                        Some(TuiMessage::System("No skills loaded.".into()))
                    } else {
                        Some(TuiMessage::System(format!(
                            "Loaded skills: {}",
                            skills.join(", ")
                        )))
                    }
                } else {
                    Some(TuiMessage::System("No skills directory found.".into()))
                }
            }

            "/compress" => {
                tracing::info!("manual compression requested");
                let _ = self.input_tx.send(AgentInput::Compress);
                Some(TuiMessage::System(
                    "Context compression triggered.".into(),
                ))
            }

            "/clear_cache" => {
                let _ = self.input_tx.send(AgentInput::Reset);
                Some(TuiMessage::System("Tool result cache cleared.".into()))
            }

            "/insights" => {
                match self.memory.status().await {
                    Ok(status) => {
                        let msg = format!(
                            "Memory -- raw: {}, compressed: {}, highest level: {}, pending: {}",
                            status.raw_count,
                            status.compressed_count,
                            status.highest_level,
                            status.pending_raw,
                        );
                        Some(TuiMessage::System(msg))
                    }
                    Err(e) => {
                        Some(TuiMessage::System(format!("Memory status error: {e}")))
                    }
                }
            }

            _ => {
                let cmds: Vec<&str> = SLASH_COMMANDS.iter().map(|&s| s).collect();
                Some(TuiMessage::System(format!(
                    "Unknown command: {}. Available: {}",
                    name,
                    cmds.join(", ")
                )))
            }
        }
    }

    pub async fn run(mut self) -> Result<(), String> {
        let mut terminal = Self::init_terminal()
            .map_err(|e| format!("terminal init failed: {e}"))?;

        {
            let mut st = self.state.lock().expect("mutex poisoned");
            st.history.push(TuiMessage::System(
                "Polymede ready. Type a message or /help for commands.".into(),
            ));
        }

        loop {
            terminal
                .draw(|frame| self.render(frame))
                .map_err(|e| format!("render error: {e}"))?;

            if event::poll(Duration::from_millis(100))
                .map_err(|e| format!("poll error: {e}"))?
            {
                if let Event::Key(key) =
                    event::read().map_err(|e| format!("event read error: {e}"))?
                {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }

                    if key.code == crossterm::event::KeyCode::Char('c')
                        && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        let mut st = self.state.lock().expect("mutex poisoned");
                        if st.is_processing {
                            st.interrupted = true;
                            st.is_processing = false;
                            st.history
                                .push(TuiMessage::System("[Interrupted]".into()));
                        } else {
                            tracing::info!("exit requested");
                            break;
                        }
                        continue;
                    }

                    if key.code == crossterm::event::KeyCode::Char('d')
                        && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        tracing::info!("exit requested");
                        break;
                    }
                }
            }

            match self.editor.readline("> ") {
                Ok(line) => {
                    let trimmed = line.trim().to_string();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let _ = self.editor.add_history_entry(line.as_str());

                    if trimmed.starts_with('/') {
                        if let Some(msg) = self.handle_slash_command(&trimmed).await {
                            self.state.lock().expect("mutex poisoned").history.push(msg);
                        }
                        continue;
                    }

                    {
                        let mut st = self.state.lock().expect("mutex poisoned");
                        st.is_processing = true;
                        st.tool_outputs.clear();
                        st.history.push(TuiMessage::UserInput(trimmed.clone()));
                    }

                    if self.input_tx.send(AgentInput::Tui(trimmed)).is_err() {
                        return Err("agent channel closed".into());
                    }
                }
                Err(_) => {
                    tracing::info!("readline error");
                    break;
                }
            }
        }

        Self::cleanup_terminal().map_err(|e| format!("terminal cleanup failed: {e}"))?;
        self.agent_shutdown.send(()).ok();

        Ok(())
    }

    fn init_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
        crossterm::terminal::enable_raw_mode()?;
        crossterm::execute!(io::stdout(), crossterm::terminal::EnterAlternateScreen)?;
        Terminal::new(CrosstermBackend::new(io::stdout()))
    }

    fn cleanup_terminal() -> io::Result<()> {
        crossterm::execute!(io::stdout(), crossterm::terminal::LeaveAlternateScreen)?;
        crossterm::terminal::disable_raw_mode()?;
        Ok(())
    }

    fn render(&self, frame: &mut Frame) {
        let st = self.state.lock().expect("mutex poisoned");
        let chunks = Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)].as_ref())
            .split(frame.size());

        let mut lines: Vec<Line> = Vec::new();

        for msg in &st.history {
            match msg {
                TuiMessage::UserInput(text) => {
                    lines.push(Line::from(vec![
                        Span::styled(
                            String::from("You: "),
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(text.clone()),
                    ]));
                }
                TuiMessage::AssistantText(text) => {
                    for line_text in text.lines() {
                        lines.push(Line::from(Span::raw(line_text.to_string())));
                    }
                }
                TuiMessage::ToolOutput { name, output, ok } => {
                    let color = if *ok { Color::Green } else { Color::Red };
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("[{}] ", name),
                            Style::default().fg(color).add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(output.clone()),
                    ]));
                }
                TuiMessage::System(text) => {
                    let styled = Span::styled(
                        format!("> {}", text),
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC),
                    );
                    lines.push(Line::from(vec![styled]));
                }
            }
        }

        if !st.current_response.is_empty() {
            for line_text in st.current_response.lines() {
                lines.push(Line::from(Span::raw(line_text.to_string())));
            }
        }

        let paragraph = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title(" Chat "))
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(Color::White));

        frame.render_widget(paragraph, chunks[0]);

        let status = format!(
            "Model: {} | Provider: {} | Tokens: {}p/{}c",
            st.model,
            st.provider,
            st.total_prompt_tokens,
            st.total_completion_tokens,
        );
        let status_bar = Paragraph::new(Line::from(Span::styled(
            status,
            Style::default().fg(Color::Gray).add_modifier(Modifier::BOLD),
        )))
        .block(Block::default().borders(Borders::ALL));

        frame.render_widget(status_bar, chunks[1]);
    }
}
