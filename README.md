# Polymede

> *Daughter of Autolycus. Raised on the lessons of her father's 12,000 commits. Armed with Khronos for time, TotalRecall for memory.*

Polymede is a self-improving AI agent built **100% in Rust** from the ground up. It inherits the architectural DNA of the Autolycus ecosystem while shedding its Python dependency, rewriting every subsystem in a language chosen for safety, speed, and fearless concurrency.

In Greek myth, the Polymedes were the wise nurses of Zeus — learned, all-knowing, the daughters of Themis (divine law). This agent is named for that legacy: a daughter of Autolycus, raised on hard-won lessons, designed to remember everything and do it right.

---

## Lineage

### Autolycus — What We Learned

Autolycus was the pioneer: the world's first AI agent for FreeBSD, a self-improving agent with 12,795 commits of battle-tested architecture. It taught us what works and what doesn't.

**Lessons carried forward:**
- The closed learning loop works — agents that create skills from experience, self-improve during use, and nudge themselves to persist knowledge outperform static agents
- Cross-platform messaging from a single gateway process is the right model
- Model-agnosticism is non-negotiable — no lock-in, no registration, no tracking
- Subagent delegation and parallelization collapse multi-step pipelines into zero-context-cost turns
- The TUI matters — multiline editing, slash-command autocomplete, streaming tool output, interrupt-and-redirect

**What we're leaving behind:**
- The Python runtime — GIL contention, dependency hell, virtual environment drift, the performance tax on every tool call
- The fragmented codebase — 78.7% Python, 12.9% TypeScript, scattered tooling that grew organically over thousands of commits
- The archived state — Autolycus has run its course. Its lessons are encoded here. Its code is not.

### Khronos — The Workflow Engine

Khronos is the lightweight durable workflow orchestration server already written in Rust. Polymede uses and evolves Khronos as its scheduling and workflow backbone:

- Cron and interval-based schedule triggers with namespace isolation
- Multi-step workflow execution with ordered activity chains
- Configurable retry policies with exponential backoff
- Heartbeat monitoring to detect and fail stalled activities
- Overlap policies: `skip`, `buffer`, or `terminate`
- Full durability — all state persisted to SQLite, survives restarts
- Clean gRPC interface: `ScheduleService`, `WorkflowService`, `WorkerService`

Polymede doesn't just consume Khronos — it extends it. The agent's own internal workflows (memory compression cycles, skill self-improvement, user-model updates) run through Khronos, making the agent's self-maintenance first-class scheduled work.

### TotalRecall — The Memory System

TotalRecall is the recursive memory compression system. Polymede integrates and re-implements its distillation pipeline in Rust:

- **Layer 0**: Raw interaction log — every command, every output, every error
- **Layer 1+**: LLM-distilled memories, recursively compressed to higher abstraction levels
- **Tag-based recall**: Retrieve relevant memories by semantic tags, bounded by token budget
- **Cross-cutting insight**: Higher compression levels surface patterns that span sessions and domains

The Rust re-implementation gains zero-copy memory access, lock-free concurrent ingestion, and the ability to run compression cycles as background tasks without blocking the main agent loop.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Polymede Agent                          │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────────┐│
│  │  TUI     │   │ Gateway  │   │     Agent Core Loop      ││
│  │  (rust-  │   │ (multi-  │   │                           ││
│  │   line)  │   │  proto)  │   │  · Turn orchestration    ││
│  └────┬─────┘   └────┬─────┘   │  · Tool dispatch         ││
│       │              │         │  · Context management    ││
│       └──────┬───────┘         │  · Skill invocation      ││
│              │                 │  · Subagent spawning     ││
│              ▼                 └──────────────┬──────────┘│
│  ┌──────────────────────────────────────────┬─┘           │
│  │         Tool Registry                    │             │
│  │  (40+ tools, extensible, zero-cost trait)│             │
│  └──────────────────────────────────────────┘             │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   Khronos   │    │ TotalRecall  │    │   User Model  │  │
│  │  (evolved)  │    │ (reimpl. in  │    │  (dialectic,  │  │
│  │             │    │   Rust)      │    │   persistent) │  │
│  │ · Scheduler │    │ · L0 ingest  │    │               │  │
│  │ · Workflows │    │ · L1+ compress│   │ · Preferences │  │
│  │ · Workers   │    │ · Tag recall  │    │ · History     │  │
│  │ · Retry     │    │ · Token budget│   │ · Beliefs     │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ SQLite   │   │  LLM     │   │   Skill   │
        │ (durability)│ (any      │   │  Store    │
        │            │  provider) │   │  (agentskills│
        └──────────┘   └──────────┘   │   compat)  │
                                      └──────────┘
```

---

## Design Principles

1. **Rust only** — No Python, no TypeScript, no polyglot drift. One language, one toolchain, one binary.
2. **Learned, not invented** — Every subsystem exists because Autolycus proved the concept. We're not guessing what works.
3. **Evolve, don't replace** — Khronos and TotalRecall are starting points, not finished products. Polymede pushes them further.
4. **No lock-in** — Any LLM provider. Any model. Switch with a command. No accounts, no tracking, no phone calls home.
5. **Durable by default** — If it can be lost, it's persisted. SQLite for state, filesystem for artifacts.
6. **Self-improving** — Skills are created from experience. Skills self-improve during use. The agent nudges itself to remember.
7. **Cross-platform** — FreeBSD, Linux, macOS. Native on all three. No emulation, no containers.

---

## Core Systems

### Agent Core Loop
The main event loop processes turns: receives input (TUI or gateway), orchestrates tool calls, manages conversation context, invokes skills, spawns subagents for parallel work, and compresses context when budgets are approached.

### Tool System
40+ tools organized in a trait-based registry. Each tool is a Rust struct implementing a common `Tool` trait. Zero-cost dispatch via dynamic trait objects. Tools include: terminal execution (via `ptyprocess`-equivalent), file operations, web search, image generation, voice, and MCP server bridging.

### Skill System
Procedural memory stored as executable skill definitions. Compatible with the agentskills.io open standard. Skills are created autonomously after complex tasks, self-improve during use, and are browsable via `/skills`.

### Memory Pipeline
TotalRecall's recursive compression, re-implemented in Rust. Raw interactions flow into Layer 0. Background Khronos workflows trigger distillation cycles that produce Layer 1+ memories. Tag-based recall with token budgeting ensures relevant context without overflow.

### User Model
Dialectic user modeling — the agent builds a deepening profile of who you are across sessions. Preferences, history, beliefs, working patterns. Persistent, private, never leaves the machine.

### Messaging Gateway
Single gateway process supporting Telegram, Discord, Slack, WhatsApp, Signal, and email. Cross-platform conversation continuity. Same slash commands, same agent brain, regardless of entry point.

### Cron & Scheduling
Khronos-evolved scheduler. Natural-language scheduled tasks with delivery to any platform. Daily reports, nightly backups, weekly audits — all running unattended.

### Subagent Delegation
Spawn isolated subagents for parallel workstreams. Each subagent gets its own context window, tool access, and result channel. Results merge back into the parent turn with zero context cost.

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Rust | Memory safety, zero-cost abstractions, fearless concurrency |
| Async runtime | Tokio | Battle-tested, ecosystem standard |
| Database | SQLite (via `sqlx`) | Durability, portability, no separate server |
| Serialization | `serde` / `serde_json` | Ergonomic, fast, ubiquitous |
| TUI | `ratatui` | Full-featured terminal UI framework |
| gRPC | `tonic` | Khronos compatibility, protobuf-native |
| LLM clients | Custom | Provider-agnostic, no lock-in |
| Logging | `tracing` | Structured, filterable, zero-overhead |
| CLI | `clap` | Derive-based, auto-generated help |
| Terminal I/O | `rustyline` + custom PTY | Multiline editing, reliable cross-platform |

---

## Getting Started

### Prerequisites

- **Rust 1.75+** (stable) — `rustup install stable`
- **SQLite3** development headers — `pkg-config --cflags sqlite3` must succeed
- **Tested platforms:**
  - FreeBSD 14.x / 15.0 (amd64)
  - Linux (amd64, aarch64)
  - macOS (x86_64, arm64)

### Build

```bash
git clone https://github.com/waym0reom3ga/Polymede.git
cd Polymede
cargo build --release
```

### First Run

```bash
# Configure API keys and model provider
./target/release/polymede setup

# Start the agent
./target/release/polymede

# Or start the messaging gateway
./target/release/polymede gateway start
```

### CLI Reference

```
polymede                     # Interactive TUI
polymede setup               # Configure providers, keys, options
polymede model               # Choose LLM provider and model
polymede tools               # Enable/disable tools
polymede gateway start       # Start messaging gateway
polymede gateway setup       # Configure messaging platforms
polymede config set <k> <v>  # Set individual config values
polymede doctor              # Diagnose issues
polymede update              # Update to latest version
```

### Slash Commands (in-session)

| Command | Description |
|---------|-------------|
| `/new` or `/reset` | Start fresh conversation |
| `/model [provider:model]` | Change model |
| `/skills` | Browse available skills |
| `/compress` | Compress context window |
| `/usage` | Show token usage stats |
| `/insights [--days N]` | Cross-session insights |
| `/stop` | Interrupt current work |

---

## Configuration

Configuration lives in `~/.config/polymede/config.toml`. All settings are documented inline. Example:

```toml
[llm]
provider = "openrouter"
model = "anthropic/claude-sonnet-4-20250514"
api_key = "..."          # or set POLYMDE_LLM_API_KEY

[llm.fallback]
provider = "lmstudio"
model = "qwen3-27b"
base_url = "http://localhost:1234/v1"

[tools]
enabled = ["terminal", "file", "web_search", "mcp"]

[memory]
compression_interval = "6h"
max_recall_tokens = 200000

[logging]
level = "info"
```

---

## Migration from Autolycus

Polymede can import your existing Autolycus configuration, memories, skills, and API keys:

```bash
polymede migrate autolycus           # Full migration
polymede migrate autolycus --dry-run # Preview what would be migrated
polymede migrate autolycus --user-data-only  # Import without secrets
```

Supported imports:
- SOUL.md persona files
- MEMORY.md and USER.md entries
- User-created skills
- Command allowlists
- Messaging platform configs
- API keys (allowlisted providers)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR process.

Quick start:

```bash
git clone https://github.com/waym0reom3ga/Polymede.git
cd Polymede
cargo build
cargo test
cargo clippy -- -D warnings
```

---

## License

LGPL v2.1 — see [LICENSE](LICENSE).

An independent project by **Technetia Inc**.  
Daughter of Autolycus. Built on lessons, not legacy.
