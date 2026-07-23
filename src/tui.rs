use std::io;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{block::Title, Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use ratatui::Terminal;
use rustyline::config::Configurer;
use rustyline::history::DefaultHistory;
use rustyline::{CompletionType, Config as RlConfig, Editor};
use tokio::sync::mpsc;
use tokio::sync::Mutex;

use crate::agent::{Agent, AgentInput, AgentOutput};
use crate::config::Config;
use crate::memory::MemoryIntegration;

// ---------------------------------------------------------------------------
// Slash commands
// ---------------------------------------------------------------------------

#[allow(dead_code)]
const SLASH_COMMANDS: &[&str] = &[
    "/new",
    "/reset",
    "/model",
    "/skills",
    "/compress",
    "/usage",
    "/insights",
    "/stop",
];

// ---------------------------------------------------------------------------
// TUI state
// ---------------------------------------------------------------------------

/// Messages displayed in the output area.
#[derive(Debug, Clone)]
enum TuiMessage {
    UserInput(String),
    AssistantText(String),
    ToolOutput { name: String, output: String, ok: bool },
    System(String),
}

/// Runtime state shared between the TUI render loop and the agent task.
struct SharedState {
    /// Accumulated assistant response for the current turn.
    current_response: String,
    /// Whether the agent is currently processing a turn.
    is_processing: bool,
    /// Tool outputs for the current turn.
    tool_outputs: Vec<TuiMessage>,
    /// Total token usage across the session.
    total_prompt_tokens: u32,
    total_completion_tokens: u32,
    /// Conversation history (user/assistant pairs).
    history: Vec<TuiMessage>,
    /// Current model name for the status bar.
    model: String,
    /// Memory status string.
    memory_status: String,
    /// Whether the user pressed Ctrl+C to interrupt.
    interrupted: bool,
}

impl SharedState {
    fn new(model: String) -> Self {
        Self {
            current_response: String::new(),
            is_processing: false,
            tool_outputs: Vec::new(),
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            history: Vec::new(),
            model,
            memory_status: "ready".into(),
            interrupted: false,
        }
    }
}

// ---------------------------------------------------------------------------
// TuiApp
// ---------------------------------------------------------------------------

pub struct TuiApp {
    /// Shared mutable state protected by a tokio Mutex.
    state: Arc<Mutex<SharedState>>,
    /// Sender for feeding input to the agent.
    input_tx: mpsc::UnboundedSender<AgentInput>,
    /// Handle to cancel the agent loop.
    agent_shutdown: mpsc::UnboundedSender<()>,
    /// rustyline editor for multiline input with history.
    editor: Editor<(), DefaultHistory>,
    /// Current model string (for slash command display).
    current_model: String,
    /// Config clone for slash command handling.
    config: Config,
    /// Memory integration for status queries.
    memory: MemoryIntegration,
}

impl TuiApp {
    /// Create a new TuiApp, spawning the agent task in the background.
    pub async fn new(config: Config) -> Result<Self, String> {
        let state_dir = Config::state_dir();

        // Memory for the agent.
        let agent_memory = MemoryIntegration::new(
            state_dir.clone(),
            config.memory.clone(),
            Some(config.llm.clone()),
            true,
        )
        .await
        .map_err(|e| format!("memory init failed: {e}"))?;

        // Separate memory instance for TUI slash commands.
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
        let shared_state = Arc::new(Mutex::new(SharedState::new(model.clone())));

        // Spawn the agent processing loop.
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

        // Build rustyline editor with history.
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

    // -----------------------------------------------------------------------
    // Output collection loop
    // -----------------------------------------------------------------------

    async fn output_loop(
        output_rx: &mut mpsc::UnboundedReceiver<AgentOutput>,
        state: &Arc<Mutex<SharedState>>,
    ) {
        loop {
            match output_rx.recv().await {
                Some(output) => {
                    let mut st = state.lock().await;
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

    /// Consume streaming text chunks and append to current_response for progressive rendering.
    async fn chunk_loop(
        chunk_rx: &mut mpsc::UnboundedReceiver<String>,
        state: &Arc<Mutex<SharedState>>,
    ) {
        loop {
            match chunk_rx.recv().await {
                Some(chunk) => {
                    let mut st = state.lock().await;
                    st.current_response.push_str(&chunk);
                }
                None => break,
            }
        }
    }

    // -----------------------------------------------------------------------
    // Slash command handling
    // -----------------------------------------------------------------------

    async fn handle_slash_command(&mut self, cmd: &str) -> Option<TuiMessage> {
        let lower = cmd.to_lowercase();

        match lower.as_str() {
            "/new" | "/reset" => {
                tracing::info!("resetting conversation");
                let mut st = self.state.lock().await;
                st.history.clear();
                st.current_response.clear();
                st.tool_outputs.clear();
                // Also clear agent-side context.
                let _ = self.input_tx.send(AgentInput::Reset);
                Some(TuiMessage::System("Conversation reset.".into()))
            }
            "/stop" => {
                tracing::info!("interrupting current turn");
                let mut st = self.state.lock().await;
                st.interrupted = true;
                st.is_processing = false;
                Some(TuiMessage::System("Turn interrupted.".into()))
            }
            "/usage" => {
                let st = self.state.lock().await;
                let msg = format!(
                    "Token usage -- prompt: {}, completion: {}, total: {}",
                    st.total_prompt_tokens,
                    st.total_completion_tokens,
                    st.total_prompt_tokens + st.total_completion_tokens,
                );
                Some(TuiMessage::System(msg))
            }
            "/model" => {
                // /model without args shows current model
                let msg = format!("Current model: {}", self.current_model);
                Some(TuiMessage::System(msg))
            }
            "/model " | "/model\t" => {
                // /model with args switches model — send to agent
                let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
                if parts.len() == 2 && !parts[1].is_empty() {
                    let new_model = parts[1].trim().to_string();
                    self.current_model = new_model.clone();
                    let _ = self.input_tx.send(AgentInput::SetModel(new_model));
                    Some(TuiMessage::System(format!("Switching model to '{}'...", parts[1].trim())))
                } else {
                    Some(TuiMessage::System(
                        "Usage: /model <name> (e.g., /model gpt-4o)".into(),
                    ))
                }
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
                Some(TuiMessage::System(format!(
                    "Unknown command: {cmd}. Available: /new /reset /model /skills /compress /usage /insights /stop"
                )))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Slash command autocomplete
    // -----------------------------------------------------------------------

    #[allow(dead_code)]
    fn slash_autocomplete(&self, line: &str) -> Option<(String, Range<usize>)> {
        if !line.starts_with('/') {
            return None;
        }

        let prefix = line.to_lowercase();
        let matches: Vec<&str> = SLASH_COMMANDS
            .iter()
            .filter(|cmd| cmd.starts_with(&prefix))
            .copied()
            .collect();

        if matches.len() == 1 {
            Some((matches[0].to_string(), 0..line.len()))
        } else if matches.len() > 1 {
            let common = Self::longest_common_prefix(&matches);
            if common.len() > prefix.len() {
                Some((common, 0..line.len()))
            } else {
                None
            }
        } else {
            None
        }
    }

    #[allow(dead_code)]
    fn longest_common_prefix(strings: &[&str]) -> String {
        if strings.is_empty() {
            return String::new();
        }
        let first = strings[0];
        let mut end = 0;
        for (i, ch) in first.char_indices() {
            let idx = first[..i].chars().count();
            if strings[1..].iter().all(|s| {
                s.chars().nth(idx).map_or(false, |c| c == ch)
            }) {
                end = i + ch.len_utf8();
            } else {
                break;
            }
        }
        first[..end].to_string()
    }

    // -----------------------------------------------------------------------
    // Run
    // -----------------------------------------------------------------------

    /// Run the TUI event loop.
    pub async fn run(mut self) -> Result<(), String> {
        let mut terminal = Self::init_terminal()
            .map_err(|e| format!("terminal init failed: {e}"))?;

        // Show welcome message.
        {
            let mut st = self.state.lock().await;
            st.history.push(TuiMessage::System(
                "Polymede ready. Type a message or /help for commands.".into(),
            ));
        }

        loop {
            // Render the UI.
            terminal
                .draw(|frame| self.render(frame))
                .map_err(|e| format!("render error: {e}"))?;

            // Check for periodic events (resize, etc).
            if event::poll(Duration::from_millis(100))
                .map_err(|e| format!("poll error: {e}"))?
            {
                if let Event::Key(key) =
                    event::read().map_err(|e| format!("event read error: {e}"))?
                {
                    // Only process key presses (not releases).
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }

                    // Ctrl+C interrupts the current turn.
                    if key.code == crossterm::event::KeyCode::Char('c')
                        && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        let mut st = self.state.lock().await;
                        if st.is_processing {
                            st.interrupted = true;
                            st.is_processing = false;
                            st.history
                                .push(TuiMessage::System("[Interrupted]".into()));
                        } else {
                            // Double Ctrl+C exits.
                            tracing::info!("exit requested");
                            break;
                        }
                        continue;
                    }

                    // Ctrl+D exits.
                    if key.code == crossterm::event::KeyCode::Char('d')
                        && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        tracing::info!("exit requested");
                        break;
                    }
                }
            }

            // Read a line from rustyline.
            match self.editor.readline("> ") {
                Ok(line) => {
                    let trimmed = line.trim().to_string();
                    if trimmed.is_empty() {
                        continue;
                    }

                    // Add to rustyline history.
                    let _ = self.editor.add_history_entry(line.as_str());

                    // Handle slash commands.
                    if trimmed.starts_with('/') {
                        if let Some(msg) = self.handle_slash_command(&trimmed).await {
                            self.state.lock().await.history.push(msg);
                        }
                        continue;
                    }

                    // Mark processing and push user input to history.
                    {
                        let mut st = self.state.lock().await;
                        st.is_processing = true;
                        st.tool_outputs.clear();
                        st.history.push(TuiMessage::UserInput(trimmed.clone()));
                    }

                    // Send input to agent.
                    if self.input_tx.send(AgentInput::Tui(trimmed)).is_err() {
                        return Err("agent input channel closed".into());
                    }
                }
                Err(rustyline::error::ReadlineError::Eof)
                | Err(rustyline::error::ReadlineError::Interrupted) => {
                    tracing::info!("input ended");
                    break;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "readline error");
                }
            }
        }

        Self::restore_terminal()
            .map_err(|e| format!("terminal restore failed: {e}"))?;
        self.agent_shutdown.send(()).ok();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Rendering
    // -----------------------------------------------------------------------

    fn render(&self, frame: &mut Frame) {
        let area = frame.area();

        // Split area: output takes most space, input hint at bottom, status bar at very bottom.
        let chunks = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(2),
            Constraint::Length(1),
        ])
        .split(area);

        // Output area.
        let state = self.state.blocking_lock();
        let output_text = Self::messages_to_lines(
            &state.history,
            &state.tool_outputs,
            &state.current_response,
        );

        let title_line = Line::from(Span::styled(
            " Conversation ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));
        let output_block = Block::default()
            .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT)
            .title(Title::from(title_line));

        let output_paragraph = Paragraph::new(output_text)
            .block(output_block)
            .wrap(Wrap { trim: true });

        frame.render_widget(output_paragraph, chunks[0]);

        // Input hint area.
        let hint = if state.is_processing {
            Line::from(Span::styled(
                "Processing... Press Ctrl+C to interrupt",
                Style::default().fg(Color::Yellow),
            ))
        } else {
            Line::from(Span::styled(
                "Ready. Type message or / for commands",
                Style::default().fg(Color::Green),
            ))
        };

        let input_block = Block::default()
            .borders(Borders::TOP | Borders::LEFT | Borders::RIGHT);

        frame.render_widget(
            Paragraph::new(hint).block(input_block),
            chunks[1],
        );

        // Status bar.
        let status_text = Self::status_bar(&state);
        let status_block = Block::default()
            .borders(Borders::LEFT | Borders::RIGHT | Borders::BOTTOM)
            .style(Style::default().bg(Color::DarkGray));

        frame.render_widget(
            Paragraph::new(status_text).block(status_block),
            chunks[2],
        );
    }

    fn messages_to_lines<'a>(
        history: &'a [TuiMessage],
        tool_outputs: &'a [TuiMessage],
        current_response: &'a str,
    ) -> Vec<Line<'a>> {
        // Helper to create a styled line from owned content.
        let styled_line = |text: String, style: Style| -> Line<'a> {
            Line::from(Span::styled(text, style))
        };
        let mut lines = Vec::new();

        for msg in history {
            match msg {
                TuiMessage::UserInput(text) => {
                    lines.push(Line::from(Span::styled(
                        "You: ",
                        Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::BOLD),
                    )));
                    lines.push(Line::from(text.clone()));
                    lines.push(Line::from(""));
                }
                TuiMessage::AssistantText(text) => {
                    lines.push(Line::from(Span::styled(
                        "Polymede: ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )));
                    for para in text.split("\n\n") {
                        for line in para.lines() {
                            lines.push(Line::from(line.to_string()));
                        }
                        lines.push(Line::from(""));
                    }
                }
                TuiMessage::ToolOutput { name, output, ok } => {
                    let color = if *ok { Color::Green } else { Color::Red };
                    lines.push(styled_line(
                        format!("[Tool: {name}]"),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ));
                    for line in output.lines() {
                        lines.push(styled_line(
                            format!("  {line}"),
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                    lines.push(Line::from(""));
                }
                TuiMessage::System(text) => {
                    lines.push(styled_line(
                        format!("* {text}"),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::ITALIC),
                    ));
                    lines.push(Line::from(""));
                }
            }
        }

        // Show current streaming response if processing.
        if !current_response.is_empty() {
            lines.push(Line::from(Span::styled(
                "Polymede: ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
            for line in current_response.lines() {
                lines.push(Line::from(line.to_string()));
            }
            lines.push(Line::from(Span::styled(
                "...",
                Style::default().fg(Color::Yellow),
            )));
        }

        // Show tool outputs for current turn.
        for msg in tool_outputs {
            if let TuiMessage::ToolOutput { name, output, ok } = msg {
                let color = if *ok { Color::Green } else { Color::Red };
                lines.push(styled_line(
                    format!("[Tool: {name}]"),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ));
                for line in output.lines() {
                    lines.push(styled_line(
                        format!("  {line}"),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
                lines.push(Line::from(""));
            }
        }

        lines
    }

    fn status_bar<'a>(state: &'a SharedState) -> Line<'a> {
        let total = state.total_prompt_tokens + state.total_completion_tokens;
        let memory_str = if state.memory_status.is_empty() {
            "memory: ready"
        } else {
            &state.memory_status
        };

        let model_text = format!(" model: {} ", state.model);
        let tokens_text = format!(" tokens: {} ", total);
        let mem_text = format!(" {memory_str}");

        Line::from(vec![
            Span::styled(
                model_text,
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                tokens_text,
                Style::default().fg(Color::Green),
            ),
            Span::styled(
                mem_text,
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                " [Ctrl+C exit] ",
                Style::default().fg(Color::DarkGray),
            ),
        ])
    }

    // -----------------------------------------------------------------------
    // Terminal setup/teardown
    // -----------------------------------------------------------------------

    fn init_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
        crossterm::terminal::enable_raw_mode()?;
        crossterm::execute!(
            io::stdout(),
            crossterm::terminal::EnterAlternateScreen,
            crossterm::cursor::Hide,
        )?;

        Terminal::new(CrosstermBackend::new(io::stdout()))
    }

    fn restore_terminal() -> io::Result<()> {
        crossterm::terminal::disable_raw_mode()?;
        crossterm::execute!(
            io::stdout(),
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::cursor::Show,
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Cleanup on drop (best effort)
// ---------------------------------------------------------------------------

impl Drop for TuiApp {
    fn drop(&mut self) {
        let _ = Self::restore_terminal();
    }
}
