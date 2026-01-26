# MultiBot 1.0 by Underground AI

A local-first, multi-agent LLM "living room" with explicit routing and persistent state.

## What This Is

MultiBot is a lightweight, browser-based app for running multiple large-language-model (LLM) agents together in a shared **living room**, where memory, identity, and communication rules persist over time.

Unlike typical "multi-agent demos," MultiBot makes **routing explicit and enforceable**: you can see, edit, and audit who can talk to whom — live — using either commands or an interactive graph editor.

MultiBot is designed for the open-source / hobbyist ecosystem and is **Ollama-first**, running locally on your home computer with no cloud dependency unless you explicitly choose one.

## Screenshots

### Empty start (no living rooms created)
<img width="1142" height="797" alt="multibot-0" src="https://github.com/user-attachments/assets/e62436ab-f0f4-4bf5-a4e2-c787f815e6ff" />

### Living room with an AI agent drafting this README document
<img width="1357" height="794" alt="MultiBot-1" src="https://github.com/user-attachments/assets/c89e57e3-5cca-4e0a-90b8-277f0128d311" />

### Living room with Advanced Options open
<img width="1143" height="822" alt="multibot-3" src="https://github.com/user-attachments/assets/348a2090-2b24-4aea-8e0f-597b4bc810f1" />

### Link graph viewer/editor
<img width="1149" height="777" alt="multibot-5" src="https://github.com/user-attachments/assets/75f52112-e15a-4b6b-bc97-80f08eb24bbf" />

### Running a MultiBot program (text adventure)
<img width="1146" height="774" alt="multibot-6" src="https://github.com/user-attachments/assets/452004c7-930c-42d9-8174-db1379c087a9" />

## Core Concepts

### Living Rooms (Persistent Environments)

Each living room is a durable workspace stored on disk. A room includes:
- Full chat transcript (`chat_history.jsonl`)
- Agent roster with models and memories (`participants.json`)
- Routing state and configuration (`config.json`)
- Shared global memory (`global_memory.txt`)
- Private user notebook (`user_notebook.txt`)
- System prompt override (`system_prompt_override.txt`)
- Snapshots and configs in subdirectories

Rooms persist across restarts and can be snapshotted or reloaded.

### Agents

Agents are named participants bound to specific models. Each agent can have:
- Optional "memories" injected into its system prompt
- Optional per-agent decode overrides
- Independent participation in routing rules

Agents may be renamed or removed without erasing historical transcript lines.

### Global Memory (Room)

A shared text buffer visible to all agents on every turn (when enabled). Useful for:
- Shared assumptions and world-building
- Group goals and project briefs
- Persistent facts and constraints

### User Notebook (Private)

A per-room notebook saved to disk that is **never sent to agents** unless you paste it manually. Perfect for:
- Planning and meta-notes
- Passwords and API keys (though we recommend a proper vault)
- Reminders to yourself

### Routing & Isolation  

MultiBot enforces **who can see whom** at the transcript and context-construction level.

**Routing Commands:**
- `/unlink BotA BotB` — prevent two agents from seeing each other
- `/link BotA BotB` — restore communication
- `/unlink all` — isolate all agents
- `/link all` — fully connect all agents
- `/links` — show current routing matrix

If two agents are unlinked, they **cannot see or reference each other's messages**, even indirectly. Routing rules are enforced consistently during:
- Context assembly
- Transcript visibility
- @mention invitations

### Interactive Routing Graph

MultiBot includes a live, interactive routing editor at:

`http://127.0.0.1:8080/graph`

In the graph:
- Nodes represent agents
- Edges represent communication links
- Clicking an edge toggles link/unlink
- "Unlink All" and "Link All" apply global isolation or restoration

Edits apply immediately and **persist to disk**. The graph stays synchronized with the room's actual routing state.

Command-line routing commands remain fully supported alongside the graph.

### Chaining via @mentions (Guarded)

Agents may invite other agents using `@AgentName`, but:
- Routing rules are enforced (no invitation across unlinked pairs)
- Invite depth is limited (prevents infinite recursion)
- Isolation is respected (unlinked agents cannot be mentioned)

This prevents accidental context leakage while enabling dynamic conversation flow.

### `.txt` Attachments (≤ ~512 KB)

Uploading a `.txt` file appends its contents to the **next user message only**. Useful for:
- Logs and debug output
- Code snippets
- Documents for discussion

---

## Installation

### Requirements

- Python 3.10 or newer
- Ollama running locally or reachable over LAN
- Python packages: `flask`, `requests`

### Setup (Recommended: Virtual Environment)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install flask requests
```

---

## Running MultiBot

Start the server with `python3 multibot.py`.

Open your browser to `http://127.0.0.1:8080`.

To change the port, run with `PORT=8080 python3 multibot.py`.

---

## Ollama Setup

### Local Ollama (Default)

MultiBot connects to Ollama at `OLLAMA_HOST`, which defaults to `http://127.0.0.1:11434`.

Click "re-scan models" in the UI to populate the model list.

### Ollama on Another Machine (LAN)

Set `OLLAMA_HOST` to the LAN machine's address:
```bash
export OLLAMA_HOST=http://192.168.1.50:11434
python3 multibot.py
```

### Ollama API Reference

See [https://docs.ollama.com/api/introduction](https://docs.ollama.com/api/introduction)

---

## Core Chat Commands

* `/stop` — stop generation
* `/reset` — start a fresh chain
* `/save [name]` — save snapshot
* `/load <file>` — load snapshot (fresh room only)
* `/saveconfig [name]` — save session config
* `/loadconfig <file>` — load session config

Routing commands:

* `/unlink BotA BotB`
* `/link BotA BotB`
* `/unlink all`
* `/link all`
* `/links`

---

## System Prompt Override (Advanced)

Each room has a built-in system prompt, but you may override it:

1. Open Advanced Options
2. Edit "System Prompt Override"
3. Save

The override is:
- Stored on disk with the room
- Included in `/saveconfig` bundles

Clearing the box restores the default prompt.

---

## Persistence and File Layout

All data lives under `ENV_DIR` (default: `envs/`).

Each room directory contains:
```
room_name/
├── chat_history.jsonl          # Append-only transcript
├── participants.json           # Agents, models, memories
├── config.json                 # Room settings, routing state
├── global_memory.txt           # Shared memory
├── user_notebook.txt           # Private notes
├── system_prompt_override.txt  # Optional override
├── snapshots/                  # Saved snapshots
└── configs/                    # Saved configs
```

Global folders `_snapshots/` and `_configs/` are also maintained.

---

## Troubleshooting

**No models listed**
- Confirm Ollama is running and reachable at `OLLAMA_HOST`.
- Check that `http://OLLAMA_HOST/api/tags` responds.

**`/load` fails**
- Snapshots load only into a fresh room. Use `/reset` first.

**Routing confusion**
- Use `/links` or open the routing graph.
- Remember: unlink rules are enforced strictly at the context level.

**Port already in use**
```bash
PORT=8081 python3 multibot.py
```

**Large attachments fail**
- Ensure file is ≤ 512 KB
- Check file extension is `.txt`
- Verify disk space in `envs/` directory

---

## MultiBot Programs

MultiBot rooms and configurations can function as reusable programs.
Examples include:

- AI-moderated text adventures
- Collaborative world-building environments
- Multi-agent research sandboxes
- Teaching demos and classroom simulations

Programs are typically shared as room snapshots or config bundles and can be loaded into fresh rooms.

---

## Design Philosophy

MultiBot is intentionally:

- **Local-first**: No cloud dependency unless you choose it
- **Explicit**: Routing, memory, and state are first-class concepts
- **Inspectable**: All state is human-readable on disk
- **Human-legible**: The system is designed to be understood, not just used

Routing, memory, and state are not hidden abstractions — they are **first-class concepts** you can see, edit, and audit.

MultiBot is meant not just for experimentation, but for thinking carefully about how multi-agent systems should live alongside people.

---

## Developers
**MultiBot 1.0** is developed by **Underground AI** for the open-source AI community. It is programmed by **Shane Oberloier** and **Robert Schneider**, with multi-agent orchestration designed by **Maxwell Schneider**, and coding assistance from a number of LLMs including **OpenAI's ChatGPT 5.1/5.2** and open-source **GPT-OSS** models, **AliBaba's** open-source **Qwen3**, and **Moonshot's** open-source **Kimi-K2**. 

**Underground AI** is a **collective** of programmers, artists, engineers, mathematicians, writers, musicians, hackers, and **dreamers.** We believe artificial intelligence is a **public resource** that should be **freely available, clean, ethical, sustainable, community based,** and should **return benefit** and **provide substantial support** to the **creative, cultural and  coding communities** whose **ideas** are the **heart of AI.**

---

## Version

MultiBot 1.0 (Beta)
