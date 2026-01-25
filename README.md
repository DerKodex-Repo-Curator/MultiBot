# MultiBot 1.0

*A local-first, multi-agent LLM living room with explicit routing and persistent state*

**MultiBot** is a lightweight, retro green-terminal web app for running multiple large-language-model agents together in a shared **living room**, where memory, identity, and communication rules persist over time.

Unlike typical “multi-agent demos,” MultiBot makes **routing explicit and enforceable**: you can see, edit, and audit who can talk to whom — live — using either commands or an interactive graph editor.

MultiBot is designed for the open-source / hobbyist ecosystem and is **Ollama-first**, running locally by default with no cloud dependency unless you explicitly choose one.

---

## What MultiBot is (in one sentence)

> **MultiBot lets you host multiple AI agents in a persistent shared room where communication rules are explicit, inspectable, and durable — and where the system stays legible to humans.**

---

## Core concepts

### Living rooms (persistent environments)

Each living room is a durable workspace stored on disk. A room includes:

* a full chat transcript
* an agent roster (names and models)
* routing state (who is linked or unlinked)
* per-room settings and advanced options
* shared global memory
* a private user notebook

Rooms persist across restarts and can be snapshotted or reloaded.

---

### Agents

Agents are named participants bound to specific models.

Each agent can have:

* optional “memories” injected into its system prompt
* optional per-agent decode overrides
* independent participation in routing rules

Agents may be renamed or removed without erasing historical transcript lines.

---

### Global Memory (room)

A shared text buffer visible to all agents on every turn (when enabled).

This is useful for:

* shared assumptions
* group goals
* persistent facts

---

### User Notebook (private)

A per-room notebook saved to disk that is **never sent to agents** unless you paste it manually.

Useful for planning, reminders, or meta-notes.

---

### Routing & isolation (the “routing law”)

MultiBot enforces **who can see whom** at the transcript and context-construction level.

Supported commands:

* `/unlink BotA BotB`
* `/link BotA BotB`
* `/unlink all`
* `/link all`
* `/links`

If two agents are unlinked, they **cannot see or reference each other’s messages**, even indirectly.

Routing rules are enforced consistently during:

* context assembly
* transcript visibility
* @mention invitations

---

### Interactive routing graph

MultiBot includes a live, interactive routing editor at:

`http://127.0.0.1:8080/graph`

In the graph:

* nodes represent agents
* edges represent communication links
* clicking an edge toggles link/unlink
* “Unlink All” and “Link All” apply global isolation or restoration

Edits apply immediately and **persist to disk**.
The graph stays synchronized with the room’s actual routing state.

Command-line routing commands remain fully supported alongside the graph.

---

### Chaining via @mentions (guarded)

Agents may invite other agents using `@AgentName`, but:

* routing rules are enforced
* invite depth is limited
* isolation is respected

This prevents accidental context leakage.

---

### `.txt` attachments (≤ ~512 KB)

Uploading a `.txt` file appends its contents to the **next user message only**.

Useful for logs, notes, or documents.

---

## Snapshots and configs

MultiBot supports two persistence modes:

* `/save [name]`
  Saves a snapshot including transcript and room state.

* `/saveconfig [name]`
  Saves a session configuration (agents, routing, memory, prompt overrides) **without transcript**.

Loading:

* `/load <file>` restores a snapshot into a fresh room (use `/reset` first).
* `/loadconfig <file>` restores a session config and starts with a clean transcript.

---

## Installation

### Requirements

* Python 3.10 or newer
* Ollama running locally or reachable over LAN
* Python packages: `flask`, `requests`

---

### Setup (recommended: virtual environment)

Create and activate a virtual environment using `python3 -m venv .venv`, then install dependencies with `pip install flask requests`.

---

## Running MultiBot

Start the server with `python3 multibot.py`.

Open your browser to `http://127.0.0.1:8080`.

To change the port, run with `PORT=8080 python3 multibot.py`.

---

## Ollama setup

### Local Ollama (default)

MultiBot connects to Ollama at `OLLAMA_HOST`, which defaults to `http://127.0.0.1:11434`.

Click “re-scan models” in the UI to populate the model list.

---

### Ollama on another machine (LAN)

Set `OLLAMA_HOST` to the LAN machine’s address, for example:

`OLLAMA_HOST=http://192.168.1.50:11434`

---

### Ollama API reference

See `https://docs.ollama.com/api/introduction`.

---

## Core chat commands

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

* open Advanced Options
* edit “System Prompt Override”
* save

The override is:

* stored on disk with the room
* included in `/saveconfig` bundles

Clearing the box restores the default prompt.

---

## Persistence and file layout

All data lives under `ENV_DIR` (default: `envs/`).

Each room directory contains:

* `chat_history.jsonl` — append-only transcript
* `participants.json` — agents, models, memories
* `config.json` — room settings including routing state
* `global_memory.txt` — shared memory
* `user_notebook.txt` — private notes
* `system_prompt_override.txt` — optional override
* `snapshots/` and `configs/` subdirectories

Global folders `_snapshots/` and `_configs/` are also maintained.

---

## Troubleshooting

**No models listed**
Confirm Ollama is running and reachable at `OLLAMA_HOST`.
Check that `http://OLLAMA_HOST/api/tags` responds.

**`/load` fails**
Snapshots load only into a fresh room. Use `/reset` first.

**Routing confusion**
Use `/links` or open the routing graph.
Remember: unlink rules are enforced strictly at the context level.

---

## Design philosophy

MultiBot is intentionally:

* local-first
* explicit
* inspectable
* human-legible

Routing, memory, and state are not hidden abstractions — they are **first-class concepts**.

MultiBot is meant not just for experimentation, but for thinking carefully about how multi-agent systems should live alongside people.
