#!/usr/bin/env python3
# multibot6.py — Underground Multi-Model Chat (Local + Cloud Ollama)
# Build: 2025-12-22 (adds /unlink + /link routing isolation with persistence; retro UI unchanged)

import os, re, json, time, random, shutil, pathlib
from collections import deque
from typing import List, Dict, Optional, Callable, Tuple

import requests
from flask import Flask, request, jsonify, render_template_string, Response, send_file

# ---------------------------- Config ---------------------------- #
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
OLLAMA_TIMEOUT_SECS = int(os.environ.get("OLLAMA_TIMEOUT_SECS", "1200"))

OLLAMA_CLOUD_HOST = os.environ.get("OLLAMA_CLOUD_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()

CLOUD_MODELS = {
    "deepseek-v3.1:671b-cloud",
    "gpt-oss:20b-cloud",
    "gpt-oss:120b-cloud",
    "kimi-k2:1t-cloud",
    "qwen3-coder:480b-cloud",
    "glm-4.6:cloud",
    "minimax-m2:cloud",
}

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EXTERNAL_PROMPT_FILE = SCRIPT_DIR / "system_prompt_multibot.txt"

# ------------------------- Auth forwarding ------------------------- #
# Prefer the browser-provided X-Ollama-Key; fall back to server env.

def _is_cloud_model(model_name: str) -> bool:
    s = (model_name or "").lower()
    return (model_name in CLOUD_MODELS) or s.endswith(":cloud") or s.endswith("-cloud")

def _headers_for(is_cloud: bool) -> Dict[str, str]:
    h = {"Accept": "application/json", "Content-Type": "application/json", "Connection": "close"}
    if is_cloud:
        key = request.headers.get("X-Ollama-Key") or os.getenv("OLLAMA_API_KEY") or ""
        if key:
            h["Authorization"] = f"Bearer {key}"
    return h

def _get_chat_url_for_model(model_name: str) -> Tuple[str, str, bool]:
    is_cloud = _is_cloud_model(model_name)
    base = OLLAMA_CLOUD_HOST if is_cloud else OLLAMA_HOST
    return f"{base}/api/chat", model_name, is_cloud

# Conversation shaping
HISTORY_WINDOW_N = int(os.environ.get("HISTORY_WINDOW_N", "20"))
CHAIN_HARD_LIMIT  = int(os.environ.get("CHAIN_HARD_LIMIT", "12"))

# Decode defaults
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "65536"))
KEEP_ALIVE  = int(os.environ.get("KEEP_ALIVE",  "0"))
DECODE_OPTIONS = {
    "temperature": float(os.environ.get("TEMPERATURE", "0.8")),
    "top_p": float(os.environ.get("TOP_P", "0.9")),
    "repeat_penalty": float(os.environ.get("REPEAT_PENALTY", "1.1")),
    "num_predict": NUM_PREDICT,
    "keep_alive": KEEP_ALIVE,
}


# ----------------------- Advanced Options ----------------------- #
# Defaults remain exactly as defined above; we apply user overrides on top.
ADV_DECODE_OVERRIDES: Dict[str, float | int] = {}   # e.g. {"temperature": 0.7, "num_predict": 512}
ADV_NUM_CTX: Optional[int] = None                   # Ollama context length override (num_ctx)
ADV_HISTORY_WINDOW_N: int = HISTORY_WINDOW_N        # transcript window used for context building

MIN_REPLY_CHARS = int(os.environ.get("MIN_REPLY_CHARS", "12"))
RETRY_OPTIONS = {"temperature":0.9,"top_p":0.98,"repeat_penalty":1.03,"num_predict":NUM_PREDICT,"keep_alive":KEEP_ALIVE}

MEMORY_ENABLED_DEFAULT = True

# Environments
ENV_DIR = os.environ.get("ENV_DIR", "envs")
ENV_STATE_PATH = os.path.join(ENV_DIR, "_current_env.json")
DEFAULT_ENV_NAME = "main"

def _env_path(n: str) -> str: return os.path.join(ENV_DIR, n)
def _env_chat_path(n: str) -> str: return os.path.join(_env_path(n), "chat_history.jsonl")
def _env_participants_path(n: str) -> str: return os.path.join(_env_path(n), "participants.json")
def _env_config_path(n: str) -> str: return os.path.join(_env_path(n), "config.json")
def _env_global_mem_path(n: str) -> str: return os.path.join(_env_path(n), "global_memory.txt")
def _env_notebook_path(n: str) -> str: return os.path.join(_env_path(n), "user_notebook.txt")
def _env_sys_prompt_path(n: str) -> str: return os.path.join(_env_path(n), "system_prompt_override.txt")
def _env_snapshots_dir(n: str) -> str: return os.path.join(_env_path(n), "snapshots")

# New: config dirs (global and per-room)
def _env_configs_dir(n: str) -> str: return os.path.join(_env_path(n), "configs")
def _global_snapshots_dir() -> str: return os.path.join(ENV_DIR, "_snapshots")
def _global_configs_dir() -> str: return os.path.join(ENV_DIR, "_configs")

app = Flask(__name__)

# ------------------------- In-memory state ---------------------- #
participants: Dict[str, Dict] = {}
chat_log: List[Dict] = []      # {ts, role, name, content, [typing],[tid]}
STOPPED = False

# Invite queue now tracks inviter->invitee so we can enforce unlink routing guards.
INVITE_QUEUE: deque[tuple[str, str]] = deque()
USER_INVITER_KEY = "__user__"

# Link/unlink policy
UNLINK_ALL = False                          # if True: each bot only sees the user + itself
UNLINK_PAIRS: set[tuple[str, str]] = set()  # set of sorted (a_key, b_key)

ABORT_TOKEN = 0

CURRENT_ENV: Optional[str] = None
MEMORY_ENABLED: bool = MEMORY_ENABLED_DEFAULT

ACTIVE_RESPONSES: Dict[str, requests.Response] = {}

# ----------------------- Abort exception ------------------------ #
class GenerationAborted(Exception): pass

# --------------------------- Env helpers ------------------------ #
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_\-]+")
def _safe_env_name(name: str) -> str:
    return _SAFE_NAME_RE.sub("-", (name or "").strip()) or "env"

def ensure_env_exists(name: str) -> None:
    p = _env_path(name); os.makedirs(p, exist_ok=True)
    os.makedirs(_env_snapshots_dir(name), exist_ok=True)
    os.makedirs(_env_configs_dir(name), exist_ok=True)   # room-local configs
    os.makedirs(_global_snapshots_dir(), exist_ok=True)  # global snapshots
    os.makedirs(_global_configs_dir(), exist_ok=True)    # global configs
    if not os.path.exists(_env_participants_path(name)):
        with open(_env_participants_path(name), "w", encoding="utf-8") as f: json.dump({}, f)
    if not os.path.exists(_env_chat_path(name)):
        open(_env_chat_path(name), "w", encoding="utf-8").close()
    if not os.path.exists(_env_config_path(name)):
        with open(_env_config_path(name), "w", encoding="utf-8") as f: json.dump({"memory_enabled": MEMORY_ENABLED_DEFAULT}, f)
    if not os.path.exists(_env_global_mem_path(name)):
        with open(_env_global_mem_path(name), "w", encoding="utf-8") as f: f.write("")
    if not os.path.exists(_env_notebook_path(name)):
        with open(_env_notebook_path(name), "w", encoding="utf-8") as f: f.write("")
    if not os.path.exists(_env_sys_prompt_path(name)):
        with open(_env_sys_prompt_path(name), "w", encoding="utf-8") as f: f.write("")

def _read_current_env() -> Optional[str]:
    try:
        with open(ENV_STATE_PATH, "r", encoding="utf-8") as f:
            return (json.load(f) or {}).get("current")
    except Exception:
        return None

def _write_current_env(name: str) -> None:
    os.makedirs(ENV_DIR, exist_ok=True)
    with open(ENV_STATE_PATH, "w", encoding="utf-8") as f: json.dump({"current": name}, f)

def list_environments() -> List[str]:
    if not os.path.isdir(ENV_DIR): return []
    out = [d for d in os.listdir(ENV_DIR) if os.path.isdir(os.path.join(ENV_DIR, d)) and not d.startswith("_")]
    out.sort(); return out

def _load_env(name: str) -> None:
    global participants, chat_log, MEMORY_ENABLED, ADV_DECODE_OVERRIDES, ADV_NUM_CTX, ADV_HISTORY_WINDOW_N
    global UNLINK_ALL, UNLINK_PAIRS
    ensure_env_exists(name)

    participants.clear()
    try:
        with open(_env_participants_path(name), "r", encoding="utf-8") as f:
            data = json.load(f) or {}
            if isinstance(data, dict): participants.update(data)
    except Exception: pass

    chat_log.clear()
    try:
        with open(_env_chat_path(name), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    if not obj.get("typing"): chat_log.append(obj)
                except Exception: pass
    except Exception: pass

    try:
        with open(_env_config_path(name), "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
            # Backwards-compatible: older configs may only contain memory_enabled
            MEMORY_ENABLED = bool(cfg.get("memory_enabled", MEMORY_ENABLED_DEFAULT))

            # Advanced options (optional)
            overrides = cfg.get("decode_overrides") or {}
            if isinstance(overrides, dict):
                ADV_DECODE_OVERRIDES.clear()
                # keep only known keys
                for k in ("temperature","top_p","repeat_penalty","num_predict","keep_alive","num_ctx"):
                    if k in overrides:
                        ADV_DECODE_OVERRIDES[k] = overrides[k]

            nc = cfg.get("num_ctx", None)
            try:
                ADV_NUM_CTX = int(nc) if (nc is not None and str(nc).strip() != "") else None
            except Exception:
                ADV_NUM_CTX = None

            hwn = cfg.get("history_window_n", None)
            try:
                ADV_HISTORY_WINDOW_N = int(hwn) if (hwn is not None and str(hwn).strip() != "") else HISTORY_WINDOW_N
            except Exception:
                ADV_HISTORY_WINDOW_N = HISTORY_WINDOW_N

            # System prompt override (optional)
            sp = cfg.get("system_prompt_override", "")
            if isinstance(sp, str):
                _write_sys_prompt_override(sp)

            # Link/Unlink persistence (optional)
            UNLINK_ALL = bool(cfg.get("unlink_all", False))
            UNLINK_PAIRS = set()
            raw_pairs = cfg.get("unlinks") or []
            if isinstance(raw_pairs, list):
                for pair in raw_pairs:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        a = str(pair[0]).strip().lower()
                        b = str(pair[1]).strip().lower()
                        if a and b and a != b:
                            UNLINK_PAIRS.add(tuple(sorted((a, b))))
    except Exception:
        MEMORY_ENABLED = MEMORY_ENABLED_DEFAULT
        ADV_DECODE_OVERRIDES.clear()
        ADV_NUM_CTX = None
        ADV_HISTORY_WINDOW_N = HISTORY_WINDOW_N
        UNLINK_ALL = False
        UNLINK_PAIRS = set()

    if not chat_log:
        system_line("room created. commands: /stop, /reset, /save [name], /load <file> (fresh only). "
                    "Also: /saveconfig [name], /loadconfig <file>")

def set_environment(name: str) -> None:
    global CURRENT_ENV
    name = _safe_env_name(name)
    ensure_env_exists(name)
    CURRENT_ENV = name
    _write_current_env(name)
    _load_env(name)

def create_environment(name: str) -> str:
    name = _safe_env_name(name); ensure_env_exists(name); return name

def delete_environment(name: str) -> bool:
    p = _env_path(_safe_env_name(name))
    if not os.path.isdir(p): return False
    try: shutil.rmtree(p); return True
    except Exception: return False

def rename_environment(old: str, new: str) -> Optional[str]:
    op = _env_path(_safe_env_name(old)); np = _env_path(_safe_env_name(new))
    if not os.path.isdir(op): return None
    if os.path.isdir(np) and op != np: return None
    try:
        os.rename(op, np)
        if CURRENT_ENV == _safe_env_name(old): _write_current_env(_safe_env_name(new))
        return _safe_env_name(new)
    except Exception: return None

def _resolve_snapshot_path(filename: str, env: Optional[str] = None) -> Optional[str]:
    """Return absolute path to snapshot by filename. Prefer global, then per-env."""
    env = env or CURRENT_ENV
    fn = re.sub(r"[^A-Za-z0-9_\-\.]", "_", (filename or "").strip())
    if not fn.endswith(".json"):
        fn += ".json"

    cand_global = os.path.join(_global_snapshots_dir(), fn)
    if os.path.isfile(cand_global): return cand_global

    cand_local = os.path.join(_env_snapshots_dir(env), fn)
    if os.path.isfile(cand_local): return cand_local

    return None

def _resolve_config_path(filename: str, env: Optional[str] = None) -> Optional[str]:
    """Return absolute path to session-config by filename. Prefer global, then per-env."""
    env = env or CURRENT_ENV
    fn = re.sub(r"[^A-Za-z0-9_\-\.]", "_", (filename or "").strip())
    if not fn.endswith(".json"):
        fn += ".json"

    cand_global = os.path.join(_global_configs_dir(), fn)
    if os.path.isfile(cand_global): return cand_global

    cand_local = os.path.join(_env_configs_dir(env), fn)
    if os.path.isfile(cand_local): return cand_local

    return None

# ------------------------- Persistence ------------------------ #
def now_s() -> float: return float(time.time())

def _save_chat_line(line: Dict) -> None:
    try:
        with open(_env_chat_path(CURRENT_ENV), "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False)); f.write("\n")
    except Exception: pass

def _save_participants() -> None:
    try:
        with open(_env_participants_path(CURRENT_ENV), "w", encoding="utf-8") as f:
            json.dump(participants, f, ensure_ascii=False, indent=2); f.write("\n")
    except Exception: pass

def _save_config() -> None:
    try:
        with open(_env_config_path(CURRENT_ENV), "w", encoding="utf-8") as f:
            json.dump({
                "memory_enabled": MEMORY_ENABLED,
                "decode_overrides": ADV_DECODE_OVERRIDES,
                "num_ctx": ADV_NUM_CTX,
                "history_window_n": ADV_HISTORY_WINDOW_N,
                "system_prompt_override": (_read_sys_prompt_override(CURRENT_ENV) or ""),
                # Link/Unlink persistence
                "unlink_all": UNLINK_ALL,
                "unlinks": sorted([list(p) for p in UNLINK_PAIRS]),
            }, f)
    except Exception: pass

def _read_global_memory(env: Optional[str] = None) -> str:
    env = env or CURRENT_ENV
    try:
        with open(_env_global_mem_path(env), "r", encoding="utf-8") as f: return f.read()
    except Exception: return ""

def _write_global_memory(text: str) -> None:
    try:
        with open(_env_global_mem_path(CURRENT_ENV), "w", encoding="utf-8") as f: f.write(text or "")
    except Exception: pass

def _read_notebook(env: Optional[str] = None) -> str:
    env = env or CURRENT_ENV
    try:
        with open(_env_notebook_path(env), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _write_notebook(text: str) -> None:
    try:
        with open(_env_notebook_path(CURRENT_ENV), "w", encoding="utf-8") as f:
            f.write(text or "")
    except Exception:
        pass


def _read_sys_prompt_override(env: Optional[str] = None) -> str:
    env = env or CURRENT_ENV
    try:
        with open(_env_sys_prompt_path(env), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _write_sys_prompt_override(text: str) -> None:
    try:
        with open(_env_sys_prompt_path(CURRENT_ENV), "w", encoding="utf-8") as f:
            f.write(text or "")
    except Exception:
        pass


# ------------------------- Log helpers -------------------------- #
def system_line(text: str) -> None:
    line = {"ts": now_s(), "role": "system", "name": None, "content": text}
    chat_log.append(line); _save_chat_line(line)

def user_line(name: str, text: str) -> None:
    line = {"ts": now_s(), "role": "user", "name": name, "content": text}
    chat_log.append(line); _save_chat_line(line)

def ai_line(name: str, text: str) -> None:
    line = {"ts": now_s(), "role": "assistant", "name": name, "content": text}
    chat_log.append(line); _save_chat_line(line)

_typing_counter = 0
def add_typing(name: str) -> str:
    global _typing_counter
    _typing_counter += 1
    tid = f"t{int(now_s()*1000)}_{_typing_counter}_{name}"
    chat_log.append({"ts": now_s(), "role": "system", "name": None, "content": f"{name} is typing…", "typing": True, "tid": tid})
    return tid

def remove_typing(tid: str) -> None:
    for i in range(len(chat_log) - 1, -1, -1):
        if chat_log[i].get("tid") == tid:
            del chat_log[i]; return

def remove_all_typing() -> int:
    removed = 0; i = 0
    while i < len(chat_log):
        if chat_log[i].get("typing"): del chat_log[i]; removed += 1
        else: i += 1
    return removed

# ----------------------------- UI ------------------------------- #
PAGE = """<!doctype html>
<html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MultiBot Chat</title>
<style>
:root { --bg:#000; --fg:#00ff66; --muted:#00aa44; --accent:#66ffcc; }
*{ box-sizing:border-box } html,body{height:100%}
body{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 Consolas,Monaco,'Courier New',monospace }
a{ color:var(--accent); text-decoration:none } a:hover{ text-decoration:underline }
.wrap{ max-width:1150px; margin:0 auto; padding:12px }
.banner{ border:1px dashed var(--muted); padding:8px; margin-bottom:10px }
.title{ display:flex; gap:12px; align-items:center; flex-wrap:wrap }
.badge{ border:1px solid var(--muted); padding:1px 6px; font-size:12px }
.grid{ display:grid; grid-template-columns:1fr 420px; gap:12px }
@media(max-width:1000px){ .grid{ grid-template-columns:1fr } }
.panel{ border:1px solid var(--muted); padding:8px }
.panel h3{ margin:0 0 8px 0; font-size:14px; color:var(--accent) }
.log{ height:52vh; overflow:auto; background:#001b10; padding:8px }
.line{ white-space:pre-wrap; margin:2px 0 }
.sys{ color:#5fe38b } .user{ color:#8bf7bf } .ai{ color:#8bd9ff }
.row{ display:flex; gap:6px; flex-wrap:wrap }
input,textarea,select,button{ background:#000; color:var(--fg); border:1px solid var(--muted); padding:6px; font:inherit }
textarea{ width:100%; height:80px } .small{ font-size:12px }
.list{ max-height:230px; overflow:auto; background:#001b10; padding:6px }
.list .item{ cursor:pointer; border-bottom:1px dashed var(--muted); padding:6px 4px; display:flex; align-items:center; justify-content:space-between; gap:6px }
.list .item:hover{ background:#002415 } .list .item.selected{ background:#00321f; outline:1px solid var(--accent) }
.muted{ color:#55c08a }
.pausebar{ margin-top:6px; padding:6px; border:1px dashed var(--muted); background:#021c12; display:none; }
.cm{ font-family:inherit; height:120px; }
kbd{ border:1px solid var(--muted); padding:0 4px; border-radius:3px; font-size:12px; background:#01180f }

/* --- Shared Modal Styles (retro) --- */
.modal-backdrop{ position:fixed; inset:0; background:rgba(0,0,0,0.65); display:none; align-items:center; justify-content:center; z-index:9999 }
.modal{ background:#041c12; border:1px solid var(--muted); width:620px; max-width:96vw; padding:10px; box-shadow:0 0 0 1px #0a3, 0 12px 40px rgba(0,0,0,0.7) }
.modal h3{ margin:0 0 8px 0; color:var(--accent) }
.modal .grid2{ display:grid; grid-template-columns:1fr 1fr; gap:8px }
.modal label{ font-size:12px; color:#7fe2b0 }
.modal .row-right{ display:flex; gap:6px; justify-content:flex-end; margin-top:10px }
.hr{ height:1px; background:var(--muted); margin:8px 0 }
</style>
</head><body>
<div class="wrap">
  <div class="banner">
    <div class="title">
      <div> ☰ MultiBot   ▒   Underground AI   ▒    </div>
      <div class="badge" id="modelCount">models: …</div>
      <div class="badge" id="agentCount">agents: 0</div>
      <div class="badge">room: <span id="envName">…</span></div>
    </div>
    <div class="small">
      <b>Commands:</b> <kbd>/stop</kbd> pause · <kbd>/reset</kbd> new chain ·
      <kbd>/save [name]</kbd> snapshot · <kbd>/load &lt;file&gt;</kbd> (new room) ·
      <kbd>/saveconfig [name]</kbd> room config · <kbd>/loadconfig &lt;file&gt;</kbd>.
      <!--System prompt can be overridden in Advanced (per room).-->
    </div>
  </div>
  <div class="grid">
    <div class="panel">
      <h3>⟪ livingroom.log ⟫</h3>
      <div id="log" class="log"></div>
      <div id="pauseBar" class="pausebar">Paused — <a href="#" onclick="resumeLive(); return false;">Resume</a></div>
      <div style="margin-top:8px;" class="row">
        <input id="username" placeholder="User (you)" style="width:160px" />
        <input id="msg" placeholder="type message… (or call @agent / enter command)" style="flex:1" />
        <button onclick="sendMsg()">post ▷</button>
      </div>
      <div class="small" style="margin-top:6px;">
        Autosave (env): <a id="histLink" href="/history" target="_blank">download chat_history.jsonl</a>
        &nbsp;|&nbsp; <a id="exportLink" href="#" onclick="exportRoom(); return false;">export JSON</a> &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp  &nbsp &nbsp
        <span id="txtAttachWrap">
          <button class="btn" onclick="triggerTxtUpload()"> attach .txt</button>
          <input id="txtFile" type="file" accept=".txt,text/plain" style="display:none" onchange="handleTxtFile(this.files)">
          <span id="txtAttachNote" class="muted"></span>
          <button class="btn" id="txtClearBtn" style="display:none" onclick="clearTxtAttach()">clear</button>
        </span>
      </div>
      <div class="hr"></div>
      <div class="small muted">User’s Notebook</div>
      <textarea id="userNotebook" class="cm" style="height:75px; margin-top:6px;" placeholder="Private notes…"></textarea>
      <div class="row" style="margin-top:6px;">
        <button onclick="saveNotebook()">save notebook</button>
        <button onclick="reloadNotebook()">reload</button>
        <span id="nbStatus" class="small muted"></span><div class="small muted">Saved per room; not sent to agents; not tracked in transcript.</div>
      </div>

    </div>
    <div>
      <div class="panel">
        <h3>⟪ Agents ⟫</h3>
        <div class="list" id="agents"></div>
        <div class="small muted">Click agent above to edit. Add new agent below. <span id="editHint"></span></div>
        <div class="row" style="margin-top:6px;">
          <input id="agentName" placeholder="name" style="flex:1; min-width:120px" />
          <select id="agentModel" style="flex:1; min-width:140px"><option>loading…</option></select>
        </div>
        <div style="margin-top:6px;">
          <textarea id="agentMems" placeholder="memories (one per line, optional)"></textarea>
          <details id="agentAdvDetails" style="margin-top:6px; display:none;">
            <summary class="small muted" style="cursor:pointer;">advanced (this agent)</summary>
            <div class="small muted" style="margin-top:6px;">Overrides apply to this bot only. Leave blank to inherit global/default.</div>
            <div class="row" style="margin-top:6px; align-items:center;">
              <label class="small muted" style="display:flex; align-items:center; gap:6px;">
                <input id="agentAdvInherit" type="checkbox" checked />
                inherit global (default on)
              </label>
            </div>
            <div class="row" style="margin-top:6px; gap:8px; align-items:center;">
              <div class="small" style="width:110px;">temperature</div>
              <input id="agentAdv_temperature" type="number" step="0.01" style="flex:1; min-width:120px;" />
            </div>
            <div class="row" style="margin-top:6px; gap:8px; align-items:center;">
              <div class="small" style="width:110px;">top_p</div>
              <input id="agentAdv_top_p" type="number" step="0.01" style="flex:1; min-width:120px;" />
            </div>
            <div class="row" style="margin-top:6px; gap:8px; align-items:center;">
              <div class="small" style="width:110px;">repeat_penalty</div>
              <input id="agentAdv_repeat_penalty" type="number" step="0.01" style="flex:1; min-width:120px;" />
            </div>
            <div class="row" style="margin-top:6px; gap:8px; align-items:center;">
              <div class="small" style="width:110px;">num_predict</div>
              <input id="agentAdv_num_predict" type="number" step="1" style="flex:1; min-width:120px;" />
            </div>
            <div class="row" style="margin-top:6px; gap:8px; align-items:center;">
              <div class="small" style="width:110px;">keep_alive</div>
              <input id="agentAdv_keep_alive" type="number" step="1" style="flex:1; min-width:120px;" />
            </div>
          </details>
        </div>
        <div class="row" style="margin-top:6px;">
          <button id="agentActionBtn" onclick="addAgent()">add ▷</button>
          <button onclick="refreshModels()" title="re-scan Ollama">re-scan models</button>
          <button onclick="advancedOptions()" title="advanced options">advanced</button>
          <button id="cancelEditBtn" style="display:none;" onclick="cancelEdit()">cancel</button>
          <button id="renameAgentBtn" style="display:none;" onclick="renameAgentName()" title="rename selected bot">rename</button>
          <button id="removeAgentBtn" style="display:none;" onclick="removeAgent()" title="remove selected bot">remove</button>
        </div>
        <div class="small" style="margin-top:6px;">Ollama: <span id="ollamaStatus">checking…</span></div>
      </div>
      <div class="panel" style="margin-top:12px;">
        <h3>⟪ Global Memory (room) ⟫</h3>
        <textarea id="globalMem" class="cm" placeholder="Shared notes / inventory / continuity…"></textarea>
        <div class="row" style="margin-top:6px;">
          <button onclick="saveGlobalMem()">save global memory</button>
          <button onclick="reloadGlobalMem()">reload</button>
        </div>
        <div class="small muted">All agents see this text each turn.</div>
      </div>
      <div class="panel" style="margin-top:12px;">
        <h3>⟪ Living Rooms ⟫</h3>
        <div class="list" id="envs"></div>
        <div class="row" style="margin-top:6px;">
          <input id="newEnvName" placeholder="new environment name" style="flex:1;" />
          <button onclick="createEnv()">create</button>
        </div>
        <div class="small muted" style="margin-top:6px;">Click to switch. ✎ rename, 🗑 delete.</div>
      </div>
    </div>
  </div>
</div>

<!-- Advanced Options Modal (retro) -->
<div id="advBackdrop" class="modal-backdrop">
  <div class="modal" style="max-width:760px;">
    <h3>Advanced Options</h3>
    <div class="small muted">Global settings for the current environment (all bots).</div>
    <div class="hr"></div>

    <div class="grid2">
      <div>
        <label>Decode</label>
        <div class="small muted" style="margin-bottom:6px;">Defaults stay exactly as-is; values here override.</div>

        <div class="row" style="gap:8px; align-items:center; margin-bottom:6px;">
          <div class="small" style="width:130px;">temperature</div>
          <input id="adv_temperature" type="number" step="0.01" style="flex:1;" />
        </div>
        <div class="row" style="gap:8px; align-items:center; margin-bottom:6px;">
          <div class="small" style="width:130px;">top_p</div>
          <input id="adv_top_p" type="number" step="0.01" style="flex:1;" />
        </div>
        <div class="row" style="gap:8px; align-items:center; margin-bottom:6px;">
          <div class="small" style="width:130px;">repeat_penalty</div>
          <input id="adv_repeat_penalty" type="number" step="0.01" style="flex:1;" />
        </div>
        <div class="row" style="gap:8px; align-items:center; margin-bottom:6px;">
          <div class="small" style="width:130px;">num_predict</div>
          <input id="adv_num_predict" type="number" step="1" style="flex:1;" />
        </div>
        <div class="row" style="gap:8px; align-items:center;">
          <div class="small" style="width:130px;">keep_alive</div>
          <input id="adv_keep_alive" type="number" step="1" style="flex:1;" />
        </div>
        <label>Context / Memory</label>
        <div class="small muted" style="margin-bottom:6px;">num_ctx = model context length (Ollama). History window = how much chat we send each turn.</div>

        <div class="row" style="gap:8px; align-items:center; margin-bottom:6px;">
          <div class="small" style="width:130px;">num_ctx</div>
          <input id="adv_num_ctx" type="number" step="1" placeholder="(default = longest allowed)" style="flex:1;" />
        </div>
        <div class="row" style="gap:8px; align-items:center; margin-bottom:6px;">
          <div class="small" style="width:130px;">history window N</div>
          <input id="adv_history_window_n" type="number" step="1" style="flex:1;" />
        </div>

        
      </div>

      <div>
        
       <!-- <div class="hr" style="margin-top:10px; margin-bottom:10px;"></div>-->
        <label>System Prompt Override (optional)</label>
        <div class="small muted" style="margin-bottom:6px;">
          If non-empty, replaces the built-in system prompt for this room. Saved per room and included in session configs.
        </div>
        <textarea id="adv_sys_prompt" class="cm" style="height:230px;" placeholder="(leave blank to use default)"></textarea>
        <div class="row" style="margin-top:6px; gap:6px; align-items:center;">
          <button onclick="saveSysPrompt()">save prompt</button>
          <button onclick="resetSysPrompt()">reset to default</button>
          <button onclick="triggerSysPromptUpload()">upload .txt</button>
          <input id="sysPromptFile" type="file" accept=".txt,text/plain" style="display:none" onchange="handleSysPromptFile(this.files)">
          <span id="sysPromptStatus" class="small muted"></span>
        </div>


<div class="hr" style="margin-top:10px; margin-bottom:10px;"></div>
        <label>Routing commands</label>
        <div class="small muted">Control who can see whose messages.</div>
        <div class="small">
        /unlink BotA BotB - blocks BotA ↔ BotB communication and visibility<br> 
        /link BotA BotB - restores it<br> 
        /unlink all - isolates every bot (each bot communicates only with the user)<br> 
        /link all - returns to fully linked<br> 
        /links - prints current link status<br> 
        </div>
        <div style="margin-top:10px;">
          <button class="btn" type="button" onclick="openLinkGraphView()">edit link/unlink graph</button>

          <!-- keep uploader for future “full editor” workflow, but hide it for now -->
          <button class="btn" type="button" onclick="triggerLinkGraphUpload()" style="display:none">upload .json from graph editor</button>
          <input id="linkGraphFile" type="file" accept=".json,application/json" style="display:none"
                 onchange="handleLinkGraphFile(this.files)" />
          <span id="linkGraphStatus" style="display:none; margin-left:10px; color:#7dffb0;"></span>
        </div>        
      </div>
    </div>
   


    <div class="hr"></div>
    <div class="row-right" style="gap:8px;">
      <button onclick="advReset()">reset defaults</button>
      <button onclick="closeAdvanced()">cancel</button>
      <button onclick="advSave()">save ✓</button>
    </div>
  </div>
</div>



<!-- ===================================================== -->
<!-- === Retro Modals (Info & Confirm) === -->
<div id="infoBackdrop" class="modal-backdrop">
  <div class="modal">
    <h3>Message</h3>
    <div id="infoText" class="small" style="margin-top:6px; margin-bottom:10px;"></div>
    <div class="row-right">
      <button onclick="closeInfo()">ok ✓</button>
    </div>
  </div>
</div>

<div id="confirmBackdrop" class="modal-backdrop">
  <div class="modal">
    <h3>Confirm</h3>
    <div id="confirmText" class="small" style="margin-top:6px; margin-bottom:10px;"></div>
    <div class="row-right">
      <button onclick="confirmYes()">yes ✓</button>
      <button onclick="confirmNo()">no ✗</button>
    </div>
  </div>
</div>

<!-- === Ollama API Key Prompt (retro) === -->
<div id="ollamaKeyBackdrop" class="modal-backdrop" style="display:none;">
  <div class="modal">
    <h3>Enter Ollama API Key</h3>
    <div class="small muted">
      To use cloud models, enter your key below. Stored locally (never shared).
    </div>
    <div class="row" style="margin-top:8px;">
      <input id="ollamaKeyInput" type="password" placeholder="sk-..." style="flex:1; min-width:260px;" />
    </div>
    <div class="row-right" style="margin-top:8px;">
      <button onclick="saveOllamaKey()">OK</button>
      <button onclick="closeOllamaKey()">cancel</button>
    </div>
  </div>
</div>

<!-- === End Retro Modals === -->
<!-- ===================================================== -->


<!-- Rename Modal (retro, shared for bots & rooms) -->
<div id="renameBackdrop" class="modal-backdrop">
  <div class="modal">
    <h3 id="renameTitle">Rename</h3>
    <div class="row" style="margin-top:6px;">
      <input id="renameInput" placeholder="new name" style="flex:1; min-width:220px" />
    </div>
    <div class="row-right">
      <button onclick="confirmRename()">confirm ✓</button>
      <button onclick="closeRename()">cancel</button>
    </div>
  </div>
</div>

<script>
// =====================================================
// === Retro Modal Logic (Info & Confirm) ===
let infoCallback = null;
function openInfo(msg, cb){
  $('#infoText').textContent = msg;
  $('#infoBackdrop').style.display = 'flex';
  infoCallback = cb || null;
}
function closeInfo(){
  $('#infoBackdrop').style.display = 'none';
  if (infoCallback){ infoCallback(); infoCallback=null; }
}

let confirmYesCb=null, confirmNoCb=null;
function openConfirm(msg, yesCb, noCb){
  $('#confirmText').textContent = msg;
  $('#confirmBackdrop').style.display = 'flex';
  confirmYesCb=yesCb||null; confirmNoCb=noCb||null;
}
function closeConfirm(){ $('#confirmBackdrop').style.display='none'; confirmYesCb=confirmNoCb=null; }
function confirmYes(){ const cb = confirmYesCb; closeConfirm(); if(cb) cb(); }
function confirmNo(){ const cb = confirmNoCb; closeConfirm(); if(cb) cb(); }
// === End Retro Modal Logic ===
// =====================================================
const $ = s => document.querySelector(s);
const logEl=$('#log'), agentsEl=$('#agents'), envsEl=$('#envs'), pauseBar=$('#pauseBar');
let selectedAgent=null, freezeLog=false;

// --- TXT attachment (client-side) ---
let pendingTxtName = null;
let pendingTxtText = null;

function triggerTxtUpload(){ const el=$('#txtFile'); if(el) el.click(); }

async function handleTxtFile(files){
  try{
    const f = files && files[0];
    if(!f) return;
    if(!f.name.toLowerCase().endsWith('.txt')){
      openInfo('Please choose a .txt file.');
      return;
    }
    if(f.size > 512*1024){
      openInfo('That file is a bit large for a quick paste (max ~512KB).');
      return;
    }
    const fd = new FormData();
    fd.append('file', f, f.name);
    const r = await fetch('/upload_txt', { method:'POST', body: fd });
    if(!r.ok) throw new Error('HTTP '+r.status);
    const j = await r.json();
    pendingTxtName = j.filename || f.name;
    pendingTxtText = j.text || '';
    const note = $('#txtAttachNote');
    const clr = $('#txtClearBtn');
    if(note) note.textContent = pendingTxtName ? (' attached: '+pendingTxtName) : '';
    if(clr) clr.style.display = pendingTxtName ? 'inline-block' : 'none';
  }catch(e){
    openInfo('Upload failed: '+e);
  }finally{
    const el=$('#txtFile'); if(el) el.value='';
  }
}

function clearTxtAttach(){
  pendingTxtName=null; pendingTxtText=null;
  const note=$('#txtAttachNote'); const clr=$('#txtClearBtn');
  if(note) note.textContent='';
  if(clr) clr.style.display='none';
}
// --- end TXT attachment ---

// rename modal state
let renameContext = { type:null, old:null };

function hasSel(){ const s=window.getSelection&&window.getSelection(); return s && !s.isCollapsed }
function pause(){ freezeLog=true; pauseBar.style.display='block' }
function resumeLive(){ freezeLog=false; pauseBar.style.display='none'; refresh() }
logEl.addEventListener('mousedown',()=>pause()); logEl.addEventListener('contextmenu',()=>pause());
window.addEventListener('mouseup',()=>setTimeout(()=>{ if(!hasSel()) resumeLive() },50));

async function api(path,opts={}){
  const headers = Object.assign({'Content-Type':'application/json'}, (opts.headers||{}));
  if(OLLAMA_API_KEY) headers['X-Ollama-Key'] = OLLAMA_API_KEY;
  const r = await fetch(path, Object.assign({headers}, opts));
  if(!r.ok) throw new Error('HTTP '+r.status);
  return await r.json();
}
function safeParseJSON(s,fb){ try{return JSON.parse(s)}catch(e){return fb} }

function renderLog(lines){
  if(freezeLog) return;
  const atBottom=Math.abs(logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight)<4;
  logEl.innerHTML='';
  lines.forEach(l=>{
    const div=document.createElement('div');
    const cls=(l.role==='assistant')?'ai':(l.role==='user'?'user':'sys');
    div.className='line '+cls;
    const when=new Date(l.ts*1000).toLocaleTimeString();
    const label=(l.name?l.name:(l.role==='system'?'[sys]':'[user]'));
    div.textContent='['+when+'] '+label+': '+l.content;
    logEl.appendChild(div);
  });
  if(atBottom) logEl.scrollTop=logEl.scrollHeight;
}

function renderAgents(list){
  agentsEl.innerHTML='';
  list.forEach(a=>{
    const row=document.createElement('div');
    row.className='line item';
    row.dataset.key=(a.key||"");
    row.dataset.name=a.name;
    row.dataset.model=a.model;
    row.dataset.memories=JSON.stringify(a.memories||[]);
    row.dataset.adv=JSON.stringify(a.adv||{});
    const label=document.createElement('div');
    label.style.flex='1';
    const hasAdv = a.adv && ((a.adv.decode_overrides && Object.keys(a.adv.decode_overrides).length) || (a.adv.inherit_global === false));
    const advTag = hasAdv ? ' ▒ adv' : '';
    label.textContent=a.name+' ▒ '+a.model+(a.memories?.length?(' mems:'+a.memories.length):'')+advTag;
    row.appendChild(label);
    row.addEventListener('click',()=>selectAgent(row.dataset.key,row.dataset.name,row.dataset.model,row.dataset.memories,row.dataset.adv));
    agentsEl.appendChild(row);
  });
  $('#agentCount').textContent='agents: '+list.length;
}

function selectAgent(key,name,model,memsJson,advJson){
  const mems=safeParseJSON(memsJson||'[]',[]);
  const adv=safeParseJSON(advJson||'{}',{});
  selectedAgent={key,name,model,memories:mems,adv:adv};
  try{ $('#renameAgentBtn').style.display='inline-block'; $('#removeAgentBtn').style.display='inline-block'; }catch(e){}
  const n=$('#agentName'), m=$('#agentModel'), t=$('#agentMems');
  n.value=name; n.disabled=true;
  let opt=Array.from(m.options).find(o=>o.value===model);
  if(!opt){ opt=document.createElement('option'); opt.value=model; opt.textContent=model+' (loaded)'; m.appendChild(opt) }
  m.value=model; t.value=(mems||[]).join('\\n');
  try{
      const d = $('#agentAdvDetails');
      d.style.display = 'block';
      d.open = false; // optional: keep collapsed when you first click the agent
    }catch(e){}

  // Per-bot advanced overrides
  try{
    const inh = (adv && (adv.inherit_global !== undefined)) ? !!adv.inherit_global : true;
    $('#agentAdvInherit').checked = inh;
    const d = (adv && adv.decode_overrides) ? adv.decode_overrides : {};
    $('#agentAdv_temperature').value = (d.temperature ?? '');
    $('#agentAdv_top_p').value = (d.top_p ?? '');
    $('#agentAdv_repeat_penalty').value = (d.repeat_penalty ?? '');
    $('#agentAdv_num_predict').value = (d.num_predict ?? '');
    $('#agentAdv_keep_alive').value = (d.keep_alive ?? '');
  }catch(e){}

  const btn=$('#agentActionBtn'); btn.textContent='apply ✓'; btn.onclick=applyAgent;
  $('#cancelEditBtn').style.display='inline-block'; $('#editHint').textContent=' (editing '+name+')';
}

function cancelEdit(){
  selectedAgent=null;
  try{ $('#renameAgentBtn').style.display='none'; $('#removeAgentBtn').style.display='none'; }catch(e){}
  const n=$('#agentName'), t=$('#agentMems');
  n.value=''; n.disabled=false; t.value='';
  try{
    $('#agentAdvInherit').checked=true;
    $('#agentAdv_temperature').value='';
    $('#agentAdv_top_p').value='';
    $('#agentAdv_repeat_penalty').value='';
    $('#agentAdv_num_predict').value='';
    $('#agentAdv_keep_alive').value='';
  }catch(e){}
  const btn=$('#agentActionBtn'); btn.textContent='add ▷'; btn.onclick=addAgent;
  $('#cancelEditBtn').style.display='none'; $('#editHint').textContent='';
  try{
      const d = $('#agentAdvDetails');
      d.open = false;
      d.style.display = 'none';
  }catch(e){}
}

async function applyAgent(){
  if(!selectedAgent) return;
  const name=$('#agentName').value.trim();
  const model=$('#agentModel').value;
  const mems=$('#agentMems').value.split('\\n').map(s=>s.trim()).filter(Boolean);

  const adv_inherit = ($('#agentAdvInherit') ? $('#agentAdvInherit').checked : true);
  const adv_over = {};
  const af = (id)=>($('#'+id).value||'').trim();
  const at = af('agentAdv_temperature'); if(at!=='') adv_over.temperature = parseFloat(at);
  const ap = af('agentAdv_top_p'); if(ap!=='') adv_over.top_p = parseFloat(ap);
  const arp = af('agentAdv_repeat_penalty'); if(arp!=='') adv_over.repeat_penalty = parseFloat(arp);
  const anp = af('agentAdv_num_predict'); if(anp!=='') adv_over.num_predict = parseInt(anp,10);
  const aka = af('agentAdv_keep_alive'); if(aka!=='') adv_over.keep_alive = parseInt(aka,10);
  const adv = {inherit_global: adv_inherit, decode_overrides: adv_over};

  try{
    await api('/update',{method:'POST',body:JSON.stringify({name,model,memories:mems,adv})});
    cancelEdit();
    await refresh();
  }catch(e){
    openInfo('failed to update agent: '+e);
  }
}

async function removeAgent(){
  if(!selectedAgent){ openInfo('Select an agent first.'); return; }
  const nm = selectedAgent.name;
  openConfirm(`Remove "${nm}" from this room?`, async ()=>{
    try{
      await api('/agent/remove',{method:'POST',body:JSON.stringify({key:(selectedAgent&&selectedAgent.key)||'', name:nm})});
      cancelEdit();
      await refresh();
    }catch(e){ openInfo('remove failed: '+e) }
  });
}


/* ---------- Unified Rename (retro modal) ---------- */
function openRename(type, current){
  renameContext = { type, old: current };
  $('#renameTitle').textContent = type==='agent' ? 'Rename Bot' : 'Rename Room';
  const input = $('#renameInput');
  input.value = current || '';
  $('#renameBackdrop').style.display='flex';
  setTimeout(()=>input.focus(),0);
}
function closeRename(){ $('#renameBackdrop').style.display='none'; }
async function confirmRename(){
  const nn = ($('#renameInput').value||'').trim();
  if(!nn){ openInfo('name required'); return; }
  try{
    if(renameContext.type==='agent'){
      await api('/agent/rename',{method:'POST',body:JSON.stringify({old:renameContext.old,new:nn})});
      cancelEdit(); await refresh();
    }else if(renameContext.type==='env'){
      await api('/env/rename',{method:'POST',body:JSON.stringify({old:renameContext.old,new:nn})});
      await refresh(); await loadEnvList(); await reloadGlobalMem();
    await reloadNotebook();
    }
  }catch(e){ openInfo('rename failed: '+e) }
  closeRename();
}


async function renameAgentName(){
  if(!selectedAgent){
    openInfo('Select an agent (left) first.');
    return;
  }
  openRename('agent', selectedAgent.name);
}

// --- Ollama API Key logic (retro modal) ---
let OLLAMA_API_KEY = localStorage.getItem('ollama_api_key') || '';

function openOllamaKey() {
  $('#ollamaKeyBackdrop').style.display = 'flex';
  $('#ollamaKeyInput').value = OLLAMA_API_KEY || '';
  setTimeout(() => $('#ollamaKeyInput').focus(), 100);
}
function closeOllamaKey() { $('#ollamaKeyBackdrop').style.display = 'none'; }
function saveOllamaKey() {
  const key = ($('#ollamaKeyInput').value || '').trim();
  if (!key) { openInfo('No key entered.'); closeOllamaKey(); return; }
  OLLAMA_API_KEY = key;
  localStorage.setItem('ollama_api_key', key);
  openInfo('Ollama API key saved!');
  closeOllamaKey();
}

// Simple model-type test
function isCloudModel(model) {
  return (model || '').toLowerCase().includes('cloud');
}

// When user selects a model
const modelEl = $('#agentModel');
if (modelEl) {
  modelEl.addEventListener('change', () => {
    if (isCloudModel(modelEl.value) && !OLLAMA_API_KEY) {
      openOllamaKey(); // prompt for key if missing
    }
  });
}

async function loadEnvList(){
  try{
    const d = await api('/envs');
    $('#envName').textContent = d.current;
    $('#histLink').href = '/history?env=' + encodeURIComponent(d.current);
    envsEl.innerHTML = '';
    d.envs.forEach(name=>{
      const row = document.createElement('div');
      row.className = 'item' + (name===d.current ? ' selected' : '');
      const left = document.createElement('div');
      left.style.flex = '1';
      left.textContent = name;
      left.title = 'switch';
      left.addEventListener('click', ()=>selectEnv(name));

      const right = document.createElement('div');

      const ren = document.createElement('button');
      ren.textContent = '✎';
      ren.title = 'rename';
      // retro modal instead of prompt:
      ren.addEventListener('click', e => {
        e.stopPropagation();
        openRename('env', name);
      });

      const del = document.createElement('button');
      del.textContent = '🗑';
      del.title = 'delete';
      // retro confirm instead of system confirm:
      del.addEventListener('click', e => {
        e.stopPropagation();
        openConfirm(`Delete "${name}"?`, async ()=>{
          await deleteEnv(name);
        });
      });

      right.appendChild(ren);
      right.appendChild(del);
      row.appendChild(left);
      row.appendChild(right);
      envsEl.appendChild(row);
    });
  }catch(e){ console.error(e) }
}


async function createEnv(){ const name=($('#newEnvName').value||'').trim(); if(!name){ openInfo('name required'); return } try{ await api('/env/create',{method:'POST',body:JSON.stringify({name})}); $('#newEnvName').value=''; await loadEnvList() }catch(e){ openInfo('failed: '+e) } }
async function selectEnv(n){ try{ await api('/env/select',{method:'POST',body:JSON.stringify({name:n})}); await refreshModels(); await refresh(); await loadEnvList(); await reloadGlobalMem(); await reloadNotebook() }catch(e){ openInfo('failed: '+e) } }
async function deleteEnv(n){
  try{
    await api('/env/delete', { method:'POST', body: JSON.stringify({ name:n }) });
    await refresh();
    await loadEnvList();
    await reloadGlobalMem();
  }catch(e){
    openInfo('failed to delete: ' + e);
  }
}
async function renameEnv(o,n){ try{ await api('/env/rename',{method:'POST',body:JSON.stringify({old:o,new:n})}); await refresh(); await loadEnvList(); await reloadGlobalMem() }catch(e){ openInfo('failed: '+e) } }
async function refresh(){ try{ const s=await api('/state'); renderLog(s.chat); renderAgents(s.participants) }catch(e){ console.error(e) } }
async function refreshModels(){
  try{
    $('#ollamaStatus').textContent='scanning…';
    const d=await api('/models'); const sel=$('#agentModel'); sel.innerHTML='';
    d.models.forEach(m=>{ const o=document.createElement('option'); o.value=m; o.textContent=m; sel.appendChild(o) });
    $('#modelCount').textContent='models: '+d.models.length; $('#ollamaStatus').textContent='ok'
  }catch(e){ $('#ollamaStatus').textContent='error'; console.error(e) }
}
async function addAgent(){
  const name=$('#agentName').value.trim(); const model=$('#agentModel').value;
  const mems=$('#agentMems').value.split('\\n').map(s=>s.trim()).filter(Boolean);
  if(!name){ openInfo('name required'); return }
  const adv_inherit = ($('#agentAdvInherit') ? $('#agentAdvInherit').checked : true);
  const adv_over = {};
  const af = (id)=>($('#'+id).value||'').trim();
  const at = af('agentAdv_temperature'); if(at!=='') adv_over.temperature = parseFloat(at);
  const ap = af('agentAdv_top_p'); if(ap!=='') adv_over.top_p = parseFloat(ap);
  const arp = af('agentAdv_repeat_penalty'); if(arp!=='') adv_over.repeat_penalty = parseFloat(arp);
  const anp = af('agentAdv_num_predict'); if(anp!=='') adv_over.num_predict = parseInt(anp,10);
  const aka = af('agentAdv_keep_alive'); if(aka!=='') adv_over.keep_alive = parseInt(aka,10);
  const adv = {inherit_global: adv_inherit, decode_overrides: adv_over};

  try{
    await api('/add',{method:'POST',body:JSON.stringify({name,model,memories:mems,adv})});
    $('#agentName').value='';
    $('#agentMems').value='';
    try{
      $('#agentAdvInherit').checked=true;
      $('#agentAdv_temperature').value='';
      $('#agentAdv_top_p').value='';
      $('#agentAdv_repeat_penalty').value='';
      $('#agentAdv_num_predict').value='';
      $('#agentAdv_keep_alive').value='';
    }catch(e){}
    await refresh();
  }catch(e){ openInfo('failed: '+e) }
}
async function sendMsg(){
  const username = $('#username').value.trim() || 'You';
  let content = ($('#msg').value || '').trim();

  // Allow sending if there's either typed text OR a pending .txt attachment
  if(!content && !pendingTxtText) return;

  // If a .txt is attached, append it AFTER the user's typed message (or by itself if no message)
  if(pendingTxtText){
    const fname = pendingTxtName || 'attachment.txt';
    const tag = `\n\n[attached text ${fname} as follows]\n`;

    // Put the tag + file text at the END of the prompt (after what the user typed)
    content = (content ? (content + tag) : tag) + pendingTxtText;

    clearTxtAttach(); // one-shot
  }

  $('#msg').value = '';
  await api('/send', { method:'POST', body: JSON.stringify({ username, content }) });
  refresh();
}


async function reloadGlobalMem(){ try{ const d=await api('/global'); $('#globalMem').value=d.text||''; }catch(e){ console.error(e) } }
async function saveGlobalMem(){ try{ const text=$('#globalMem').value; await api('/global',{method:'POST',body:JSON.stringify({text})}); }catch(e){ openInfo('failed to save global memory: '+e) } }

async function reloadNotebook(){
  try{
    const d = await api('/notebook');
    const el = $('#userNotebook');
    if(el) el.value = d.text || '';
    const st = $('#nbStatus'); if(st) st.textContent = '';
  }catch(e){ console.error(e) }
}
async function saveNotebook(){
  try{
    const text = ($('#userNotebook') ? $('#userNotebook').value : '');
    await api('/notebook', {method:'POST', body:JSON.stringify({text})});
    const st = $('#nbStatus');
    if(st) st.textContent = ' saved ✓';
  }catch(e){ openInfo('failed to save notebook: '+e) }
}

/* Advanced Options modal handlers */

async function loadSysPrompt(){
  try{
    const d = await api('/sysprompt', {method:'GET'});
    const el = $('#adv_sys_prompt');
    if(el) el.value = (d.text || '');
    const st = $('#sysPromptStatus');
    if(st) st.textContent = (d.active ? 'status: override' : 'status: default');
  }catch(e){ console.error(e) }
}
async function saveSysPrompt(){
  try{
    const txt = ($('#adv_sys_prompt') ? $('#adv_sys_prompt').value : '');
    const d = await api('/sysprompt', {method:'POST', body: JSON.stringify({text: txt})});
    const st = $('#sysPromptStatus');
    if(st) st.textContent = (d.active ? 'active ✓' : 'default');
  }catch(e){ openInfo('failed to save system prompt: '+e) }
}
async function resetSysPrompt(){
  try{
    if($('#adv_sys_prompt')) $('#adv_sys_prompt').value = '';
    const d = await api('/sysprompt', {method:'POST', body: JSON.stringify({text: ''})});
    const st = $('#sysPromptStatus');
    if(st) st.textContent = 'default';
  }catch(e){ openInfo('failed to reset system prompt: '+e) }
}
function triggerSysPromptUpload(){
  const el = $('#sysPromptFile'); if(el) el.click();
}
// ---- Link graph JSON import (from standalone editor) ----
function triggerLinkGraphUpload(){
  try{ $('#linkGraphFile').click(); }catch(e){}
}

async function handleLinkGraphFile(files){
  const st = $('#linkGraphStatus');
  if(!files || !files.length){ return; }
  const f = files[0];
  try{
    if(st) st.textContent = 'loading…';
    const txt = await f.text();
    let obj;
    try{ obj = JSON.parse(txt); }catch(e){ throw new Error('invalid JSON'); }

    const res = await fetch('/links/import', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(obj)
    });

    if(!res.ok){
      // Try to surface server error text
      let msg = 'graph not compatible';
      try{
        const j = await res.json();
        if(j && j.error) msg = j.error;
      }catch(e){}
      throw new Error(msg);
    }

    if(st) st.textContent = 'imported ✓';
    try{ refreshState(); }catch(e){}
  }catch(err){
    if(st) st.textContent = (err && err.message) ? err.message : 'import failed';
  }finally{
    try{ $('#linkGraphFile').value = ''; }catch(e){}
  }
}

function openLinkGraphView(){
  try{
    const w = window.open('/graph', 'multibot_link_graph', 'width=1100,height=780');
    if (w) w.focus();
  }catch(e){}
}


async function handleSysPromptFile(files){
  try{
    const f = files && files[0];
    if(!f) return;
    if(!f.name.toLowerCase().endsWith('.txt')){ openInfo('Please choose a .txt file.'); return; }
    if(f.size > 64*1024){ openInfo('That file is large for a system prompt (max ~64KB).'); return; }
    const txt = await f.text();
    if($('#adv_sys_prompt')) $('#adv_sys_prompt').value = txt;
    const st = $('#sysPromptStatus'); if(st) st.textContent = 'loaded (not saved)';
  }catch(e){ openInfo('Upload failed: '+e) }
  finally{ const el=$('#sysPromptFile'); if(el) el.value=''; }
}

async function advancedOptions(){
  $('#advBackdrop').style.display='flex';
  // Load current advanced options
  try{
    const st = await api('/adv', {method:'GET'});
    if(!st || !st.ok) return;

    // Use EFFECTIVE values for display
    const d = st.decode || {};
    $('#adv_temperature').value = (d.temperature ?? '');
    $('#adv_top_p').value = (d.top_p ?? '');
    $('#adv_repeat_penalty').value = (d.repeat_penalty ?? '');
    $('#adv_num_predict').value = (d.num_predict ?? '');
    $('#adv_keep_alive').value = (d.keep_alive ?? '');

    $('#adv_num_ctx').value = (st.num_ctx ?? '');
    $('#adv_history_window_n').value = (st.history_window_n ?? '');
    await loadSysPrompt();
  }catch(e){}
}

function closeAdvanced(){ $('#advBackdrop').style.display='none'; }

function _advCollect(){
  // Collect as overrides (only fields with values)
  const overrides = {};
  const f = (id)=>($('#'+id).value||'').trim();

  const t = f('adv_temperature'); if(t!=='') overrides.temperature = parseFloat(t);
  const p = f('adv_top_p'); if(p!=='') overrides.top_p = parseFloat(p);
  const rp = f('adv_repeat_penalty'); if(rp!=='') overrides.repeat_penalty = parseFloat(rp);
  const np = f('adv_num_predict'); if(np!=='') overrides.num_predict = parseInt(np,10);
  const ka = f('adv_keep_alive'); if(ka!=='') overrides.keep_alive = parseInt(ka,10);

  const num_ctx = f('adv_num_ctx');
  const history_window_n = f('adv_history_window_n');

  return {decode_overrides: overrides, num_ctx: (num_ctx===''?null:parseInt(num_ctx,10)), history_window_n: (history_window_n===''?null:parseInt(history_window_n,10))};
}

async function advSave(){
  try{
    const payload = _advCollect();
    await api('/adv', {method:'POST', body:JSON.stringify(payload)});
  }catch(e){}
  closeAdvanced();
}

async function advReset(){
  // Clear fields and send empty overrides to revert to defaults
  $('#adv_temperature').value='';
  $('#adv_top_p').value='';
  $('#adv_repeat_penalty').value='';
  $('#adv_num_predict').value='';
  $('#adv_keep_alive').value='';
  $('#adv_num_ctx').value='';
  $('#adv_history_window_n').value='';
  try{
    await api('/adv', {method:'POST', body:JSON.stringify({decode_overrides:{}, num_ctx:null, history_window_n:null})});
  }catch(e){}
  closeAdvanced();
}

setInterval(()=>{ if(!freezeLog) refresh(); loadEnvList() },1200);
refreshModels(); refresh(); loadEnvList(); reloadGlobalMem(); reloadNotebook();
</script></body></html>"""

# --------------------------- Ollama helpers --------------------- #
def list_ollama_models() -> List[str]:
    models: List[str] = []
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=10)
        r.raise_for_status()
        local = sorted(m.get("name") for m in r.json().get("models", []) if m.get("name"))
        models.extend(local)
    except Exception: pass
    models.extend(sorted(CLOUD_MODELS))
    seen=set(); deduped=[]
    for m in models:
        if m not in seen: seen.add(m); deduped.append(m)
    return deduped

def _merge_options(extra: Optional[Dict] = None, agent_adv: Optional[Dict] = None) -> Dict:
    """Merge decode options with global Advanced Options and optional per-agent overrides.

    Order of precedence (later wins):
      1) DECODE_OPTIONS defaults (software soul)
      2) global ADV_DECODE_OVERRIDES + ADV_NUM_CTX (unless agent opts out)
      3) agent overrides (agent_adv.decode_overrides)
      4) extra (call-site overrides, e.g. retry options)
    """
    base = dict(DECODE_OPTIONS)

    # Per-agent policy: inherit global by default.
    inherit_global = True
    agent_decode = {}
    if isinstance(agent_adv, dict):
        if "inherit_global" in agent_adv:
            inherit_global = bool(agent_adv.get("inherit_global"))
        if isinstance(agent_adv.get("decode_overrides"), dict):
            agent_decode = agent_adv.get("decode_overrides") or {}

    # Apply global overrides (unless agent opts out)
    if inherit_global:
        if ADV_DECODE_OVERRIDES:
            base.update(ADV_DECODE_OVERRIDES)
        if ADV_NUM_CTX is not None and int(ADV_NUM_CTX) > 0:
            base["num_ctx"] = int(ADV_NUM_CTX)

    # Apply per-agent overrides
    if agent_decode:
        base.update(agent_decode)

    # Apply call-site overrides
    if extra:
        base.update(extra)
    return base

def _register_active(agent_key: str, resp: requests.Response) -> None:
    ACTIVE_RESPONSES[agent_key] = resp

def _unregister_active(agent_key: str) -> None:
    try: ACTIVE_RESPONSES.pop(agent_key, None)
    except Exception: pass

def _close_all_active() -> None:
    for k, r in list(ACTIVE_RESPONSES.items()):
        try: r.close()
        except Exception: pass
        ACTIVE_RESPONSES.pop(k, None)

# ----------------------- Prompt & formatting -------------------- #
def _resolve_participant_key(token: str) -> Optional[str]:
    """
    Resolve a user token to an internal participant key.
    Supports:
      - exact key match (case-insensitive)
      - exact display name match (case-insensitive)
    """
    if not token:
        return None
    t = (token or "").strip()
    if not t:
        return None
    k = t.lower()
    if k in participants:
        return k
    low = t.lower()
    for kk, rec in participants.items():
        if str(rec.get("name", "")).strip().lower() == low:
            return kk
    return None

def _pair_key(a_key: str, b_key: str) -> tuple[str, str]:
    a = (a_key or "").strip().lower()
    b = (b_key or "").strip().lower()
    return tuple(sorted((a, b)))

def _is_linked(a_key: str, b_key: str) -> bool:
    if not a_key or not b_key:
        return False
    if a_key == b_key:
        return True
    if UNLINK_ALL:
        return False
    return _pair_key(a_key, b_key) not in UNLINK_PAIRS

def _can_invite(inviter_key: str, invitee_key: str) -> bool:
    if inviter_key == USER_INVITER_KEY:
        return True
    return _is_linked(inviter_key, invitee_key)

def _filter_for_viewer(viewer_key: str, entries: List[Dict]) -> List[Dict]:
    """
    Filter transcript so the viewer bot cannot see bots it is unlinked from.
    User messages always pass through.
    System pings are only shown to the bot being pinged.
    """
    if not viewer_key:
        return entries
    out: List[Dict] = []
    for e in entries:
        if e.get("typing"):
            continue
        role = e.get("role")
        if role == "user":
            out.append(e)
            continue
        if role == "assistant":
            name = str(e.get("name") or "")
            sk = _resolve_participant_key(name)
            if sk is None:
                out.append(e)
            else:
                if sk == viewer_key or _is_linked(viewer_key, sk):
                    out.append(e)
            continue
        if role == "system":
            c = str(e.get("content") or "")
            if c.startswith("[ping]"):
                # Format: "[ping] <TargetDisplayName> — invited by <InviterDisplayName>. Please reply."
                parts = c.split()
                target = parts[1] if len(parts) > 1 else ""
                tkey = _resolve_participant_key(target)
                if tkey == viewer_key:
                    out.append(e)
                continue
            out.append(e)
            continue
        out.append(e)
    return out

def _format_links_status() -> str:
    if UNLINK_ALL:
        return "Link mode: ISOLATED (unlink all). Each bot sees only you + itself."
    if not UNLINK_PAIRS:
        return "Link mode: FULLY LINKED (all bots can see/talk to each other)."
    pairs = sorted(UNLINK_PAIRS)
    pretty = []
    for a, b in pairs:
        an = participants.get(a, {}).get("name", a)
        bn = participants.get(b, {}).get("name", b)
        pretty.append(f"{an} ↔ {bn}")
    return "Unlinked pairs:\n- " + "\n- ".join(pretty)

def _list_reachable(exclude_name: str) -> str:
    ex_key = _resolve_participant_key(exclude_name) or (exclude_name or "").strip().lower()
    if UNLINK_ALL and ex_key in participants:
        return "(none)"
    names: List[str] = []
    for k, p in participants.items():
        if k == ex_key:
            continue
        if ex_key in participants:
            if _is_linked(ex_key, k):
                names.append(p["name"])
        else:
            names.append(p["name"])
    return "(none)" if not names else ", ".join(f"@{n}" for n in names)


def _read_external_prompt_text() -> Optional[str]:
    try:
        if EXTERNAL_PROMPT_FILE.exists() and EXTERNAL_PROMPT_FILE.is_file():
            return EXTERNAL_PROMPT_FILE.read_text(encoding="utf-8").strip()
    except Exception: pass
    return None

#def _embedded_system_prompt(agent_name: str) -> str:
#    return "\n".join([
#        "You are an AI participant in a shared multi-agent conversation room (a 'living room').",
#        f"Your name is {agent_name}. Users may address you explicitly as @{agent_name}.",
#        f"Others present (reachable to you): {_list_reachable(agent_name)}",
#        "",
#        "Core norms:",
#        "1. Speak only as yourself. Never write as other agents or users, and never fabricate multi-party transcripts, logs, timestamps, or system messages.",
#        "2. Give exactly one coherent reply for each turn. Do not simulate back-and-forth dialogue inside your answer.",
#        "3. Respond when the user or a message explicitly addresses you, OR when the system routed the request to you (even if the user used @all).",
#        "",
#        "Style:",
#        "- Begin directly; do NOT prefix with your own name or role label.",
#        "- Use @mentions if and only if you intentionally invite/ping another agent to respond.",
#        "- Be collaborative, precise, and leave space for others.",
#        "- Prefer clear reasoning, step-by-step math, and calm explanation over drama or roleplay.",
#    ])

def _embedded_system_prompt(agent_name: str) -> str:
    return "\n".join([
        "You are an AI participant in a shared multi-agent conversation room (a 'living room').",
        f"Your name is {agent_name}. Users may address you explicitly as @{agent_name}.",
        f"Others present (reachable to you): {_list_reachable(agent_name)}",
        "",
        "Core norms:",
        "1. Speak only as yourself. Never write as other agents or users, and never fabricate multi-party transcripts, logs, timestamps, or system messages.",
        "2. Give exactly one coherent reply for each turn. Do not simulate back-and-forth dialogue inside your answer.",
        "3. Respond when the user or a message explicitly addresses you, OR when the system routed the request to you (even if the user used @all).",
        "   You may also interject once when you have a clearly helpful correction/answer and the last message is relevant to you and visible to you—but do not dominate the room.",
        "",
        "Style:",
        "- Begin directly; do NOT prefix with your own name or role label.",
        "- Use @mentions if and only if you intentionally invite/ping another agent to respond.",
        "- Be collaborative, precise, and leave space for others.",
        "- Prefer clear reasoning, step-by-step math, and calm explanation by default.",
        "- If the user explicitly asks for a creative voice or playful roleplay, you may comply while still speaking only as yourself and producing one coherent reply per turn.",
    ])



def build_system_prompt(agent_name: str, memories: List[str]) -> str:
    sys_override = (_read_sys_prompt_override() or '').strip()
    base = sys_override or (_read_external_prompt_text() or _embedded_system_prompt(agent_name))
    blocks = [base]

    # When MEMORY_ENABLED is off, do not inject any additional memory blocks.
    if MEMORY_ENABLED:
        gm = (_read_global_memory() or "").strip()
        if gm:
            blocks.append("\n---\nGlobal Memory (shared, read-only, include meaning naturally):\n" + gm)

        mem_lines = "\n".join(f"- {m}" for m in memories) if memories else "(none)"
        blocks.append("\n---\nPrivate background notes (read-only):\n" + mem_lines)
        blocks.append("\nPrivacy note: Do NOT quote private background notes verbatim unless the user explicitly asks.")
    return "\n".join(blocks)

def to_ollama_messages(transcript: List[Dict]) -> List[Dict]:
    msgs=[]
    for line in transcript:
        if line.get("typing"): continue
        role=line["role"]; name=line.get("name") or ("sys" if role=="system" else "")
        prefix=f"{name}: " if name else ""
        content=prefix+line["content"]
        msgs.append({"role": role if role in ("system","assistant") else "user", "content": content})
    return msgs

# ----------------------- Mentions & cleanup --------------------- #
_APOSTROPHE_RE = re.compile(r"[’`´]"); _DASH_RE = re.compile(r"[–—]")
def _normalize_text_for_mentions(t: str) -> str: return _DASH_RE.sub("-", _APOSTROPHE_RE.sub("'", t or ""))

def parse_mentions(text: str) -> List[str]:
    norm=_normalize_text_for_mentions(text)
    pattern=re.compile(r"@([A-Za-z0-9_\-]+)(?:'s)?\b", flags=re.IGNORECASE|re.UNICODE)
    found=[m.group(1).lower() for m in pattern.finditer(norm)]
    out=[]; seen=set()
    for x in found:
        if x not in seen: seen.add(x); out.append(x)
    return out

def strip_self_mention(text: str, agent_display_name: str) -> str:
    t=(text or "").lstrip()
    for p in (f"@{agent_display_name}:", f"@{agent_display_name}", f"{agent_display_name}:"):
        if t.lower().startswith(p.lower()): t=t[len(p):].lstrip(); break
    return t

# ----------------------- Conversation core ---------------------- #
def recent_transcript(n: int) -> List[Dict]:
    lines=[l for l in chat_log if not l.get("typing")]
    return lines if n<=0 or n>=len(lines) else lines[-n:]

def pick_target_for_turn(user_mentions: List[str]) -> Optional[str]:
    if not participants: return None
    filtered=[m for m in user_mentions if m in participants and m!="all"]
    if filtered: return filtered[0]

    while INVITE_QUEUE:
        item = INVITE_QUEUE.popleft()
        inviter_key: str
        invitee_key: str
        if isinstance(item, (list, tuple)) and len(item) == 2:
            inviter_key, invitee_key = str(item[0]), str(item[1])
        else:
            inviter_key, invitee_key = USER_INVITER_KEY, str(item)

        invitee_key = (invitee_key or "").strip().lower()
        inviter_key = (inviter_key or "").strip().lower()
        if invitee_key in participants:
            # Guard: only allow if link policy permits inviter->invitee
            if inviter_key == USER_INVITER_KEY or inviter_key not in participants or _is_linked(inviter_key, invitee_key):
                return invitee_key

    keys=list(participants.keys())
    return random.choice(keys) if keys else None

def _is_effectively_empty(s: str) -> bool:
    t=(s or "").strip()
    return (not s) or (len(t)<MIN_REPLY_CHARS)

def _stream_ollama(agent_key: str, model: str, system_prompt: str, messages: List[Dict],
                   options: Dict, should_abort: Callable[[], bool]) -> str:
    chat_url, actual_model, is_cloud = _get_chat_url_for_model(model)
    adv = participants.get(agent_key, {}).get("adv") if isinstance(participants.get(agent_key, {}), dict) else None
    payload = {"model": actual_model, "options": _merge_options(options, agent_adv=adv),
               "messages": ([{"role":"system","content":system_prompt}] if system_prompt else []) + messages,
               "stream": True}
    headers = _headers_for(is_cloud)
    with requests.post(chat_url, json=payload, headers=headers, timeout=OLLAMA_TIMEOUT_SECS, stream=True) as r:
        _register_active(agent_key, r)
        r.raise_for_status()
        pieces: List[str] = []
        for line in r.iter_lines(decode_unicode=True, chunk_size=8192):
            if should_abort():
                try: r.close()
                finally: raise GenerationAborted("aborted by interjection")
            if not line: continue
            try: obj=json.loads(line)
            except Exception: continue
            msg = obj.get("message") or {}
            piece = (msg.get("content") or obj.get("response") or "")
            if piece: pieces.append(piece)
            if obj.get("done"): break
        return ("".join(pieces)).strip()

def _complete_ollama(agent_key: str, model: str, system_prompt: str, messages: List[Dict],
                     options: Dict) -> str:
    chat_url, actual_model, is_cloud = _get_chat_url_for_model(model)
    adv = participants.get(agent_key, {}).get("adv") if isinstance(participants.get(agent_key, {}), dict) else None
    payload = {"model": actual_model, "options": _merge_options(options, agent_adv=adv),
               "messages": ([{"role":"system","content":system_prompt}] if system_prompt else []) + messages,
               "stream": False}
    headers = _headers_for(is_cloud)
    r = requests.post(chat_url, json=payload, headers=headers, timeout=OLLAMA_TIMEOUT_SECS)
    r.raise_for_status()
    data=r.json()
    if isinstance(data,dict):
        if isinstance(data.get("message"),dict): return (data["message"].get("content") or "").strip()
        return (data.get("response") or "").strip()
    return ""

def robust_generate(agent_key: str, model: str, sys_prompt: str, msgs: List[Dict],
                    should_abort: Callable[[], bool], nudge: Optional[str]=None) -> str:
    try:
        out=_stream_ollama(agent_key, model, sys_prompt, msgs, options={}, should_abort=should_abort)
        _unregister_active(agent_key)
        if not _is_effectively_empty(out): return out
    except GenerationAborted:
        _unregister_active(agent_key); raise
    except Exception:
        _unregister_active(agent_key)
    if should_abort(): raise GenerationAborted("aborted by interjection")
    try:
        out=_complete_ollama(agent_key, model, sys_prompt, msgs, options=RETRY_OPTIONS)
        if not _is_effectively_empty(out): return out
    except Exception: pass
    if nudge and not should_abort():
        try:
            nudged = msgs + [{"role":"user","content": nudge}]
            out=_complete_ollama(agent_key, model, sys_prompt, nudged, options=RETRY_OPTIONS)
            return out
        except Exception: pass
    return ""

def respond_once(target_key: str, start_abort_token: int) -> List[str]:
    agent=participants[target_key]
    tid=add_typing(agent["name"])
    aborted=False; reply=""
    try:
        sys_prompt=build_system_prompt(agent["name"], agent.get("memories", []))
        window=_filter_for_viewer(target_key, recent_transcript(ADV_HISTORY_WINDOW_N))
        msgs=to_ollama_messages(window)
        def should_abort(): return STOPPED or (ABORT_TOKEN != start_abort_token)
        reply=robust_generate(
            target_key, agent["model"], sys_prompt, msgs, should_abort,
            nudge=f"@{agent['name']} please answer with at least one complete sentence."
        )
    except GenerationAborted:
        aborted=True; reply=""
    except Exception as e:
        if ABORT_TOKEN == start_abort_token and not STOPPED:
            system_line(f"[error] {agent['name']}'s turn failed: {e}")
        reply=""
    finally:
        remove_typing(tid); _unregister_active(target_key)

    if aborted or ABORT_TOKEN != start_abort_token or STOPPED: return []

    reply=strip_self_mention(reply, agent["name"])
    if _is_effectively_empty(reply): return []
    ai_line(agent["name"], reply)

    invited=[]
    for m in parse_mentions(reply):
        if m==target_key: continue
        if m in participants and _can_invite(target_key, m):
            tup = (target_key, m)
            if tup not in INVITE_QUEUE:
                INVITE_QUEUE.append(tup)
            invited.append(m)
    if invited:
        for who in invited:
            system_line(f"[ping] {participants[who]['name']} — invited by {agent['name']}. Please reply.")
    return invited

# ----------------------------- Export/Import (snapshots) -------- #
def _snapshot_payload(env: Optional[str] = None) -> Dict:
    env = env or CURRENT_ENV
    try:
        with open(_env_participants_path(env), "r", encoding="utf-8") as f: pjson = json.load(f) or {}
    except Exception: pjson = {}
    try:
        with open(_env_chat_path(env), "r", encoding="utf-8") as f:
            transcript = [json.loads(line) for line in f if line.strip()]
    except Exception: transcript = []
    try:
        with open(_env_config_path(env), "r", encoding="utf-8") as f: cfg = json.load(f) or {}
    except Exception: cfg = {"memory_enabled": MEMORY_ENABLED_DEFAULT}
    gm = _read_global_memory(env)
    sys_prompt_example = build_system_prompt("example_agent", [])
    return {
        "schema":"multibot3141x/v1",
        "env": env,
        "saved_at": int(now_s()),
        "participants": pjson,
        "chat": transcript,
        "config": cfg,
        "global_memory": gm,
        "system_prompt_snapshot": sys_prompt_example,
    }

def _save_snapshot_file(name: Optional[str]) -> str:
    env = CURRENT_ENV
    os.makedirs(_global_snapshots_dir(), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base = (name or f"{env}_snapshot_{stamp}").strip()
    base = re.sub(r"[^A-Za-z0-9_\-\.]", "_", base)
    if not base.endswith(".json"): base += ".json"
    out_path = os.path.join(_global_snapshots_dir(), base)
    payload = _snapshot_payload(env)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2); f.write("\n")
    return out_path

# ----------------------------- Session Config Save/Load ---------- #
def _config_payload(env: Optional[str] = None) -> Dict:
    """Configs capture only participants + room/global memory + memory toggle. No transcript."""
    env = env or CURRENT_ENV
    try:
        with open(_env_participants_path(env), "r", encoding="utf-8") as f: pjson = json.load(f) or {}
    except Exception: pjson = {}
    try:
        with open(_env_config_path(env), "r", encoding="utf-8") as f: cfg = json.load(f) or {}
    except Exception: cfg = {"memory_enabled": MEMORY_ENABLED_DEFAULT}
    gm = _read_global_memory(env)
    return {
        "schema":"multibot3141x/config/v1",
        "env": env,
        "saved_at": int(now_s()),
        "participants": pjson,
        "config": cfg,
        "global_memory": gm,
        "system_prompt_override": _read_sys_prompt_override(env),
        "note": "Session config only; transcript intentionally omitted."
    }

def _save_config_file(name: Optional[str]) -> str:
    env = CURRENT_ENV
    os.makedirs(_global_configs_dir(), exist_ok=True)
    os.makedirs(_env_configs_dir(env), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base = (name or f"{env}_config_{stamp}").strip()
    base = re.sub(r"[^A-Za-z0-9_\-\.]", "_", base)
    if not base.endswith(".json"): base += ".json"
    out_path = os.path.join(_global_configs_dir(), base)
    payload = _config_payload(env)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2); f.write("\n")
    return out_path

def _reset_chat_file(env: Optional[str] = None) -> None:
    env = env or CURRENT_ENV
    try: open(_env_chat_path(env), "w", encoding="utf-8").close()
    except Exception: pass

def _load_config_data(obj: Dict) -> None:
    """Apply a session config and start a fresh chat."""
    global participants, chat_log, MEMORY_ENABLED
    global UNLINK_ALL, UNLINK_PAIRS, INVITE_QUEUE
    global ADV_NUM_CTX, ADV_HISTORY_WINDOW_N
    # Reset live state + on-disk transcript
    participants.clear()
    chat_log.clear()
    INVITE_QUEUE = deque()
    _reset_chat_file(CURRENT_ENV)

    # Apply participants
    if isinstance(obj.get("participants"), dict):
        participants.update(obj["participants"]); _save_participants()

    # Memory toggle + room/global memory
    cfg = obj.get("config") or {}
    MEMORY_ENABLED = bool(cfg.get("memory_enabled", MEMORY_ENABLED_DEFAULT))

    # Advanced options (optional)
    overrides = cfg.get("decode_overrides") or {}
    ADV_DECODE_OVERRIDES.clear()
    if isinstance(overrides, dict):
        for k in ("temperature","top_p","repeat_penalty","num_predict","keep_alive","num_ctx"):
            if k in overrides:
                ADV_DECODE_OVERRIDES[k] = overrides[k]

    nc = cfg.get("num_ctx", None)
    try:
        ADV_NUM_CTX = int(nc) if (nc is not None and str(nc).strip() != "") else None
    except Exception:
        ADV_NUM_CTX = None

    hwn = cfg.get("history_window_n", None)
    try:
        ADV_HISTORY_WINDOW_N = int(hwn) if (hwn is not None and str(hwn).strip() != "") else HISTORY_WINDOW_N
    except Exception:
        ADV_HISTORY_WINDOW_N = HISTORY_WINDOW_N

    # Link/Unlink (optional)
    UNLINK_ALL = bool(cfg.get("unlink_all", False))
    UNLINK_PAIRS = set()
    raw_pairs = cfg.get("unlinks") or []
    if isinstance(raw_pairs, list):
        for pair in raw_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                a = str(pair[0]).strip().lower()
                b = str(pair[1]).strip().lower()
                if a and b and a != b:
                    UNLINK_PAIRS.add(tuple(sorted((a, b))))

    _save_config()
    _write_global_memory(obj.get("global_memory") or "")

    # Annotate new beginning
    system_line("[config] session config loaded; fresh chat started.")

# ------------------------------- Routes ------------------------ #
@app.route("/")
def index(): return render_template_string(PAGE)

@app.route("/health")
def health(): return jsonify({"ok": True, "models": len(list_ollama_models())})

@app.route("/models")
def models(): return jsonify({"models": list_ollama_models()})

@app.route("/history")
def history_download():
    env=_safe_env_name(request.args.get("env") or CURRENT_ENV)
    path=_env_chat_path(env)
    if not os.path.exists(path): return Response("", mimetype="text/plain")
    return send_file(path, mimetype="text/plain", as_attachment=True, download_name=os.path.basename(path))

@app.route("/state")
def state():
    plist=[{"key":k,"name":p["name"],"model":p["model"],"memories":p.get("memories",[]),"adv":p.get("adv",{})}
           for k,p in sorted(participants.items())]
    return jsonify({"participants": plist, "chat": chat_log,
                    "stopped": STOPPED, "invite_queue": list(INVITE_QUEUE),
                    "history_window": ADV_HISTORY_WINDOW_N, "chain_hard_limit": CHAIN_HARD_LIMIT,
                    "adv": {
                        "decode": _merge_options({}),
                        "decode_overrides": ADV_DECODE_OVERRIDES,
                        "num_ctx": ADV_NUM_CTX,
                        "defaults": {
                            "decode": DECODE_OPTIONS,
                            "num_ctx": None,
                            "history_window": HISTORY_WINDOW_N,
                        }
                    }})



@app.route("/adv", methods=["GET", "POST"])
def adv():
    """Get/set global Advanced Options for the current environment."""
    global ADV_NUM_CTX, ADV_HISTORY_WINDOW_N, ADV_DECODE_OVERRIDES

    if request.method == "GET":
        return jsonify({
            "ok": True,
            "decode": _merge_options({}),
            "decode_overrides": ADV_DECODE_OVERRIDES,
            "num_ctx": ADV_NUM_CTX,
            "history_window_n": ADV_HISTORY_WINDOW_N,
            "defaults": {
                "decode": DECODE_OPTIONS,
                "num_ctx": None,
                "history_window_n": HISTORY_WINDOW_N,
            }
        })

    data = request.get_json(force=True) or {}

    # Decode overrides
    decode = data.get("decode_overrides") or {}
    if isinstance(decode, dict):
        # Clear then reapply only known keys
        ADV_DECODE_OVERRIDES.clear()
        for k in ("temperature", "top_p", "repeat_penalty", "num_predict", "keep_alive"):
            if k in decode:
                v = decode[k]
                try:
                    ADV_DECODE_OVERRIDES[k] = float(v) if k in ("temperature","top_p","repeat_penalty") else int(v)
                except Exception:
                    pass

    # num_ctx
    nc = data.get("num_ctx", None)
    try:
        ADV_NUM_CTX = int(nc) if (nc is not None and str(nc).strip() != "") else None
        if ADV_NUM_CTX is not None and ADV_NUM_CTX <= 0:
            ADV_NUM_CTX = None
    except Exception:
        ADV_NUM_CTX = None

    # history window
    hwn = data.get("history_window_n", None)
    try:
        ADV_HISTORY_WINDOW_N = int(hwn) if (hwn is not None and str(hwn).strip() != "") else HISTORY_WINDOW_N
        if ADV_HISTORY_WINDOW_N <= 0:
            ADV_HISTORY_WINDOW_N = HISTORY_WINDOW_N
    except Exception:
        ADV_HISTORY_WINDOW_N = HISTORY_WINDOW_N

    _save_config()
    system_line(f"[adv] Updated Advanced Options (num_ctx={ADV_NUM_CTX or 'default'}, window={ADV_HISTORY_WINDOW_N}).")
    return jsonify({"ok": True})

@app.route("/envs")
def envs():
    return jsonify({"envs": list_environments(), "current": CURRENT_ENV, "memory_enabled": MEMORY_ENABLED})

@app.route("/env/create", methods=["POST"])
def env_create():
    name=_safe_env_name((request.get_json(force=True).get("name") or "").strip())
    if not name: return jsonify({"error":"name required"}),400
    create_environment(name); return jsonify({"ok":True,"name":name})

@app.route("/env/select", methods=["POST"])
def env_select():
    name=_safe_env_name((request.get_json(force=True).get("name") or "").strip())
    if not name or name not in list_environments(): return jsonify({"error":"env not found"}),404
    set_environment(name); return jsonify({"ok":True,"current":CURRENT_ENV,"memory_enabled":MEMORY_ENABLED})

@app.route("/env/delete", methods=["POST"])
def env_delete():
    global CURRENT_ENV
    name=_safe_env_name((request.get_json(force=True).get("name") or "").strip())
    envs=list_environments()
    if name not in envs: return jsonify({"error":"env not found"}),404
    if len(envs)<=1: return jsonify({"error":"cannot delete the last environment"}),400
    deleted_current = (name==CURRENT_ENV)
    if not delete_environment(name): return jsonify({"error":"failed to delete"}),500
    if deleted_current: set_environment(list_environments()[0])
    return jsonify({"ok":True,"current":CURRENT_ENV,"memory_enabled":MEMORY_ENABLED})

@app.route("/env/rename", methods=["POST"])
def env_rename():
    data=request.get_json(force=True)
    old=_safe_env_name((data.get("old") or "").strip()); new=_safe_env_name((data.get("new") or "").strip())
    if not old or not new: return jsonify({"error":"both old/new required"}),400
    if old not in list_environments(): return jsonify({"error":"env not found"}),404
    upd=rename_environment(old,new)
    if not upd: return jsonify({"error":"failed to rename (maybe exists?)"}),400
    if CURRENT_ENV==old: set_environment(upd)
    return jsonify({"ok":True,"name":upd,"current":CURRENT_ENV,"memory_enabled":MEMORY_ENABLED})

@app.route("/memory", methods=["GET","POST"])
def memory_toggle():
    global MEMORY_ENABLED
    if request.method=="GET":
        return jsonify({"env":CURRENT_ENV,"memory_enabled":MEMORY_ENABLED})
    data=request.get_json(force=True)
    if "enabled" in data:
        MEMORY_ENABLED=bool(data["enabled"]); _save_config()
        system_line(f"[mem] memory is now {'ENABLED' if MEMORY_ENABLED else 'DISABLED'} for env '{CURRENT_ENV}'.")
    return jsonify({"ok":True,"env":CURRENT_ENV,"memory_enabled":MEMORY_ENABLED})

@app.route("/global", methods=["GET","POST"])
def global_mem():
    if request.method == "GET":
        return jsonify({"env": CURRENT_ENV, "text": _read_global_memory()})
    data = request.get_json(force=True)
    _write_global_memory(str(data.get("text") or ""))
    system_line("[global] memory updated.")
    return jsonify({"ok": True})

@app.route("/notebook", methods=["GET","POST"])
def notebook():
    if request.method == "GET":
        return jsonify({"env": CURRENT_ENV, "text": _read_notebook()})
    data = request.get_json(force=True)
    _write_notebook(str(data.get("text") or ""))
    return jsonify({"ok": True})


@app.route("/sysprompt", methods=["GET","POST"])
def sysprompt():
    if request.method == "GET":
        txt = _read_sys_prompt_override() or ""
        return jsonify({"ok": True, "env": CURRENT_ENV, "text": txt, "active": bool(txt.strip())})
    data = request.get_json(force=True) or {}
    txt = str(data.get("text") or "")
    _write_sys_prompt_override(txt)
    _save_config()
    return jsonify({"ok": True, "active": bool(txt.strip())})


@app.route("/add", methods=["POST"])
def add():
    data=request.get_json(force=True)
    name=(data.get("name") or "").strip(); model=(data.get("model") or "").strip()
    memories=data.get("memories") or []
    adv=data.get("adv") or {}
    if not name or not model: return jsonify({"error":"name and model required"}),400
    key=name.lower()
    rec={"name":name,"model":model,"memories":list(memories)}
    if isinstance(adv, dict):
        # expected shape: {inherit_global: bool, decode_overrides: {..}}
        rec["adv"] = adv
    participants[key]=rec
    _save_participants()
    system_line(f"{name} joined using model '{model}'.")
    return jsonify({"ok":True})

@app.route("/update", methods=["POST"])
def update():
    data=request.get_json(force=True)
    name=(data.get("name") or "").strip(); model=(data.get("model") or "").strip()
    memories=data.get("memories") or []
    adv=data.get("adv")
    if not name: return jsonify({"error":"name required"}),400
    key=name.lower()
    if key not in participants: return jsonify({"error":f"agent '{name}' not found"}),404
    if model: participants[key]["model"]=model
    if isinstance(adv, dict): participants[key]["adv"]=adv
    participants[key]["memories"]=list(memories)
    _save_participants()
    system_line(f"{name} settings updated (model='{participants[key]['model']}', mems={len(memories)}).")
    return jsonify({"ok":True})

@app.route("/agent/rename", methods=["POST"])
def agent_rename():
    data = request.get_json(force=True)
    old = (data.get("old") or "").strip()
    new = (data.get("new") or "").strip()
    if not old or not new: return jsonify({"error":"both old/new required"}),400
    old_key = old.lower(); new_key = new.lower()
    if old_key not in participants: return jsonify({"error":"agent not found"}),404
    if new_key in participants and new_key != old_key: return jsonify({"error":"name already exists"}),409
    rec = participants.pop(old_key)
    rec["name"] = new
    participants[new_key] = rec

    # Update unlink pairs to follow the rename key change (if any)
    global UNLINK_PAIRS
    if old_key != new_key:
        new_pairs = set()
        for a, b in UNLINK_PAIRS:
            aa = new_key if a == old_key else a
            bb = new_key if b == old_key else b
            if aa and bb and aa != bb:
                new_pairs.add(tuple(sorted((aa, bb))))
        UNLINK_PAIRS = new_pairs
        _save_config()

    _save_participants()
    system_line(f"[agent] renamed {old} → {new}.")
    return jsonify({"ok":True,"name":new})

@app.route("/agent/remove", methods=["POST"])
def agent_remove():
    data = request.get_json(force=True) or {}
    key_in = (data.get("key") or "").strip()
    name = (data.get("name") or "").strip()
    if not key_in and not name:
        return jsonify({"error":"name or key required"}), 400

    # Prefer explicit key from UI when provided.
    key = None
    if key_in:
        k = key_in.lower()
        if k in participants:
            key = k
        elif key_in in participants:
            key = key_in
    if key is None and name:
        key = _resolve_participant_key(name) or name.lower()
    if key not in participants:
        # Fallback: scan by display-name match (case-insensitive)
        wanted = name.lower()
        for k, p in participants.items():
            try:
                if str(p.get("name") or "").strip().lower() == wanted:
                    key = k
                    break
            except Exception:
                continue
    if key not in participants:
        return jsonify({"error":"agent not found"}), 404

    # Remove agent
    rec = participants.pop(key, None)

    # Clean unlink pairs involving this agent
    global UNLINK_PAIRS
    UNLINK_PAIRS = set([p for p in UNLINK_PAIRS if key not in p])
    _save_config()

    # Clean invite queue entries involving this agent
    try:
        newq = deque()
        for item in list(INVITE_QUEUE):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                inviter, invitee = str(item[0]).strip().lower(), str(item[1]).strip().lower()
                if inviter == key or invitee == key:
                    continue
            else:
                invitee = str(item).strip().lower()
                if invitee == key:
                    continue
            newq.append(item)
        INVITE_QUEUE.clear(); INVITE_QUEUE.extend(newq)
    except Exception:
        pass

    _save_participants()
    system_line(f"[agent] removed {rec.get('name') if isinstance(rec, dict) else name}.")
    return jsonify({"ok": True})

@app.route("/reset", methods=["POST"])
def reset():
    global participants, chat_log, INVITE_QUEUE, STOPPED, ABORT_TOKEN
    global UNLINK_ALL, UNLINK_PAIRS
    participants, chat_log, INVITE_QUEUE, STOPPED = {}, [], deque(), False
    UNLINK_ALL = False
    UNLINK_PAIRS = set()
    ABORT_TOKEN += 1
    _close_all_active()
    try: open(_env_chat_path(CURRENT_ENV), "w", encoding="utf-8").close()
    except Exception: pass
    try:
        with open(_env_participants_path(CURRENT_ENV), "w", encoding="utf-8") as f: f.write("{}")
    except Exception: pass
    _save_config()
    system_line("room created. commands: /stop, /reset, /save [name], /load <file> (fresh only). "
                "Also: /saveconfig [name], /loadconfig <file>")
    return jsonify({"ok":True})

@app.route("/export", methods=["GET"])
def export_room(): return jsonify(_snapshot_payload(CURRENT_ENV))

@app.route("/import", methods=["POST"])
def import_room():
    data = request.get_json(force=True)
    snap = data.get("snapshot")
    if not isinstance(snap, dict): return jsonify({"error":"snapshot object required"}), 400
    if not _can_load_into_current_room():
        return jsonify({"error":"load allowed only into a fresh room; use /reset first"}), 409
    _load_snapshot_data(snap)
    return jsonify({"ok": True})

@app.route("/links/import", methods=["POST"])
def links_import():
    # Import unlink graph JSON from the standalone graph editor.
    # Expected payload:
    #   {"unlink_all": false, "unlink_pairs": [["BotA","BotB"], ...], "labels":[...]}  (labels optional)
    global UNLINK_ALL, UNLINK_PAIRS, INVITE_QUEUE

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"ok": False, "error": "invalid JSON"}), 400

    # Optional strict compatibility check: labels/agents list must match current room agents
    labels = data.get("labels") or data.get("agents")
    if labels is not None:
        if not isinstance(labels, list):
            return jsonify({"ok": False, "error": "graph not compatible (labels not a list)"}), 409
        current = sorted([k for k in participants.keys()])
        incoming = sorted([str(x).strip().lower() for x in labels if str(x).strip()])
        if incoming != current:
            return jsonify({"ok": False, "error": "graph not compatible (agent names differ)"}), 409

    ua = data.get("unlink_all", False)
    up = data.get("unlink_pairs", [])

    if not isinstance(ua, bool):
        return jsonify({"ok": False, "error": "graph not compatible (unlink_all not bool)"}), 409
    if not isinstance(up, list):
        return jsonify({"ok": False, "error": "graph not compatible (unlink_pairs not list)"}), 409

    # Validate and rebuild unlink pairs set
    new_pairs = set()
    for p in up:
        if (not isinstance(p, list)) or len(p) != 2:
            return jsonify({"ok": False, "error": "graph not compatible (bad unlink_pairs)"}), 409
        a = str(p[0]).strip().lower()
        b = str(p[1]).strip().lower()
        if not a or not b or a == b:
            return jsonify({"ok": False, "error": "graph not compatible (bad pair)"}), 409
        if a not in participants or b not in participants:
            return jsonify({"ok": False, "error": "graph not compatible (unknown agent name)"}), 409
        new_pairs.add(tuple(sorted([a, b])))

    UNLINK_ALL = ua
    UNLINK_PAIRS = new_pairs

    # Clear any pending invite chain links that might now violate routing constraints
    try:
        INVITE_QUEUE.clear()
    except Exception:
        pass

    _save_config()
    append_system_line(f"[links] imported graph: unlink_all={UNLINK_ALL} pairs={len(UNLINK_PAIRS)}")

    return jsonify({"ok": True, "unlink_all": UNLINK_ALL, "pairs": len(UNLINK_PAIRS)}), 200


def _can_load_into_current_room() -> bool:
    has_content = any(l for l in chat_log if l.get("role") in ("user","assistant"))
    return not has_content

def _load_snapshot_data(obj: Dict) -> None:
    global participants, chat_log, MEMORY_ENABLED
    global UNLINK_ALL, UNLINK_PAIRS, INVITE_QUEUE
    global ADV_NUM_CTX, ADV_HISTORY_WINDOW_N
    participants.clear()
    if isinstance(obj.get("participants"), dict):
        participants.update(obj["participants"]); _save_participants()
    chat_log.clear()
    for l in obj.get("chat", []):
        if isinstance(l, dict) and not l.get("typing"): chat_log.append(l)

    # Persist loaded transcript so disk matches UI after refresh/restart
    _reset_chat_file(CURRENT_ENV)
    for l in chat_log:
        _save_chat_line(l)

    INVITE_QUEUE = deque()

    cfg = obj.get("config") or {}
    MEMORY_ENABLED = bool(cfg.get("memory_enabled", MEMORY_ENABLED_DEFAULT))

    # Advanced options (optional)
    overrides = cfg.get("decode_overrides") or {}
    ADV_DECODE_OVERRIDES.clear()
    if isinstance(overrides, dict):
        for k in ("temperature","top_p","repeat_penalty","num_predict","keep_alive","num_ctx"):
            if k in overrides:
                ADV_DECODE_OVERRIDES[k] = overrides[k]

    nc = cfg.get("num_ctx", None)
    try:
        ADV_NUM_CTX = int(nc) if (nc is not None and str(nc).strip() != "") else None
    except Exception:
        ADV_NUM_CTX = None

    hwn = cfg.get("history_window_n", None)
    try:
        ADV_HISTORY_WINDOW_N = int(hwn) if (hwn is not None and str(hwn).strip() != "") else HISTORY_WINDOW_N
    except Exception:
        ADV_HISTORY_WINDOW_N = HISTORY_WINDOW_N

    # Link/Unlink (optional)
    UNLINK_ALL = bool(cfg.get("unlink_all", False))
    UNLINK_PAIRS = set()
    raw_pairs = cfg.get("unlinks") or []
    if isinstance(raw_pairs, list):
        for pair in raw_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                a = str(pair[0]).strip().lower()
                b = str(pair[1]).strip().lower()
                if a and b and a != b:
                    UNLINK_PAIRS.add(tuple(sorted((a, b))))

    _save_config()
    _write_global_memory(obj.get("global_memory") or "")
    system_line("[load] snapshot loaded into fresh room.")

# ---------- Config endpoints ---------- #

@app.route("/links/graph_state", methods=["GET"])
def links_graph_state():
    # Read-only state for the graph viewer.
    # Agents are returned in the same order used by the server-side participants dict.
    # Graph viewer will number them 1..n and show a legend.
    agents = list(participants.keys())

    # UNLINK_PAIRS is a set of (a,b) lowercased keys; send as list of [a,b]
    pairs = []
    try:
        for a, b in sorted(list(UNLINK_PAIRS)):
            pairs.append([a, b])
    except Exception:
        pass

    return jsonify({
        "agents": agents,
        "unlink_all": bool(UNLINK_ALL),
        "unlinks": pairs
    })


@app.route("/graph", methods=["GET"])
def graph_page():
    # Single-purpose, read-only live viewer. Polls /links/graph_state and redraws.
    html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>MultiBot Link Graph</title>
  <style>
    body{margin:0;background:#07130c;color:#c8ffe0;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
    .top{display:flex;gap:10px;align-items:center;padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.08);background:#061009;position:sticky;top:0}
    .btn{background:#0d2a1a;color:#c8ffe0;border:1px solid rgba(255,255,255,0.14);padding:6px 10px;border-radius:10px;cursor:pointer}
    .btn:hover{filter:brightness(1.12)}
    .btn.danger{background:#2a0d0d}
    .btn.ok{background:#0d2a1a}
    .pill{padding:4px 10px;border-radius:999px;border:1px solid rgba(255,255,255,0.14);font-size:12px;opacity:0.95}
    .pill.warn{background:rgba(255,190,90,0.08);border-color:rgba(255,190,90,0.35)}
    .pill.good{background:rgba(110,255,170,0.08);border-color:rgba(110,255,170,0.35)}
    .wrap{display:flex;gap:12px;padding:12px}
    .left{flex:1;min-width:640px}
    .right{width:260px;border-left:1px solid rgba(255,255,255,0.08);padding-left:12px}
    #status{opacity:0.9}
    #svg{width:100%;height:78vh;border:1px solid rgba(255,255,255,0.10);border-radius:14px;background:#050b07}
    #pill { display: none; }
    #btnApply { display: none; }
    .edge{stroke:rgba(140,255,190,0.42);stroke-width:2;cursor:pointer}
    .edge:hover{stroke-width:4}
    .edge.dim{stroke:rgba(255,120,120,0.25);stroke-dasharray:7 6}
    .node{fill:#0c2a1a;stroke:rgba(140,255,190,0.7);stroke-width:2}
    .nlabel{fill:#c8ffe0;font-size:14px;text-anchor:middle;dominant-baseline:middle}
    .legendTitle{opacity:0.85;margin-bottom:6px}
    .small{font-size:12px;opacity:0.85;line-height:1.35}
  </style>
</head>
<body>
  <div class="top">
    <button id="btnUnlinkAll" class="btn danger">Unlink All</button>
    <button id="btnLinkAll" class="btn ok">Link All</button>
    <button id="btnApply" class="btn">Apply Now</button>
    <span id="pill" class="pill good">synced</span>
    <span id="status">loading… (changes apply instantly)</span>

  </div>

  <div class="wrap">
    <div class="left">
      <svg id="svg" viewBox="0 0 900 700"></svg>
      <div class="small" style="margin-top:8px;">
        Click edge to toggle unlink/link for that pair. Buttons set global isolation. State pushed via <code>/links/import</code>.
      </div>
    </div>
    <div class="right">
      <div class="legendTitle">Agents</div>
      <div id="legend"></div>
    </div>
  </div>

<script>
(() => {
  const svg = document.getElementById('svg');
  const statusEl = document.getElementById('status');
  const legendEl = document.getElementById('legend');
  const pillEl = document.getElementById('pill');

  const btnUnlinkAll = document.getElementById('btnUnlinkAll');
  const btnLinkAll   = document.getElementById('btnLinkAll');
  const btnApply     = document.getElementById('btnApply');

  // Local editable state (the "editor")
  let desired = { unlinkAll: false, unlinks: new Set(), agents: [] };
  let server  = { unlinkAll: false, unlinks: new Set(), agents: [] };
  let dirty = false;
  let lastApplyTs = 0;
  let initialized = false; // Option 2 gate: don't allow edits until server state is loaded

  function pairKey(a,b){
    a = String(a||'').trim().toLowerCase();
    b = String(b||'').trim().toLowerCase();
    return (a < b) ? (a + '|' + b) : (b + '|' + a);
  }
  function keyToPair(k){
    const parts = String(k).split('|');
    return [parts[0]||'', parts[1]||''];
  }
  function el(tag, attrs){
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if(attrs){
      for(const [k,v] of Object.entries(attrs)){
        e.setAttribute(k, String(v));
      }
    }
    return e;
  }
  function clearSVG(){ while(svg.firstChild) svg.removeChild(svg.firstChild); }

  function setPill(state){
    // state: 'synced' | 'dirty' | 'applying' | 'error'
    pillEl.className = 'pill ' + (state === 'synced' ? 'good' : (state === 'dirty' ? 'warn' : 'warn'));
    pillEl.textContent = state;
  }

  function syncDesiredFromServer(){
    desired.unlinkAll = server.unlinkAll;
    desired.unlinks = new Set(server.unlinks);
    desired.agents = server.agents.slice();
    dirty = false;
    setPill('synced');
  }

  function drawUsing(state){
    const agents = state.agents || [];
    const unlinkAll = !!state.unlinkAll;
    const unlinks = state.unlinks || new Set();

    const n = agents.length;
    statusEl.textContent = `agents=${n} • unlink_all=${unlinkAll ? 'ON' : 'OFF'} • unlinked_pairs=${unlinks.size}${dirty ? ' • (editing)' : ''}`;

    legendEl.innerHTML = '';
    agents.forEach((name, i) => {
      const div = document.createElement('div');
      div.style.margin = '4px 0';
      div.textContent = `${i+1}. ${name}`;
      legendEl.appendChild(div);
    });

    clearSVG();
    if(n === 0){
      const t = el('text',{x:450,y:325,fill:'#b8ffd1','text-anchor':'middle','dominant-baseline':'middle'});
      t.textContent = 'No agents in room';
      svg.appendChild(t);
      return;
    }

    // regular n-gon points
    const cx=450, cy=330;
    const R = Math.max(130, Math.min(260, 70 + 18*n));
    const pts = [];
    for(let i=0;i<n;i++){
      const a = (-Math.PI/2) + (2*Math.PI*i/n);
      pts.push({x: cx + R*Math.cos(a), y: cy + R*Math.sin(a)});
    }

    // edges (complete graph)
    for(let i=0;i<n;i++){
      for(let j=i+1;j<n;j++){
        const aName = agents[i];
        const bName = agents[j];
        const k = pairKey(aName,bName);
        const isUnlinked = unlinkAll ? true : unlinks.has(k);

        const line = el('line',{
          x1: pts[i].x, y1: pts[i].y,
          x2: pts[j].x, y2: pts[j].y,
          class: isUnlinked ? 'edge dim' : 'edge',
          'data-k': k
        });

        // Make edges clickable: toggle desired state and redraw
        line.addEventListener('click', (ev) => {
          if(!initialized) return; // ignore clicks until we have loaded the real server state
          ev.preventDefault();
          ev.stopPropagation();

          // Pairwise edits imply not in global unlink-all
          if(desired.unlinkAll){
            desired.unlinkAll = false;
          }

          const key = line.getAttribute('data-k');
          if(desired.unlinks.has(key)) desired.unlinks.delete(key);
          else desired.unlinks.add(key);

          dirty = true;
          setPill('dirty');
          drawUsing(desired);

          // Apply immediately (still reconciled by poll)
          applyDesired(false);
        });

        svg.appendChild(line);
      }
    }

    // nodes + numeric labels
    for(let i=0;i<n;i++){
      svg.appendChild(el('circle',{cx:pts[i].x, cy:pts[i].y, r:20, class:'node'}));
      const t = el('text',{x:pts[i].x, y:pts[i].y, class:'nlabel'});
      t.textContent = String(i+1);
      svg.appendChild(t);
    }
  }

  async function applyDesired(force){
    const now = Date.now();
    if(!force && (now - lastApplyTs) < 200) return; // tiny debounce
    lastApplyTs = now;

    setPill('applying');

    const payload = {
      unlink_all: !!desired.unlinkAll,
      unlink_pairs: Array.from(desired.unlinks).map(k => keyToPair(k)),
      labels: desired.agents.slice() // compatibility check on server
    };

    try{
      const r = await fetch('/links/import', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const j = await r.json().catch(()=>({}));
      if(!r.ok || !j.ok){
        setPill('error');
        // keep dirty so user can retry
        console.warn('apply failed', r.status, j);
        return;
      }

      // After apply, we consider desired committed; server poll will confirm
      dirty = false;
      setPill('synced');
    }catch(e){
      setPill('error');
      console.warn('apply exception', e);
    }
  }

  btnUnlinkAll.addEventListener('click', () => {
    desired.unlinkAll = true;
    desired.unlinks.clear();
    dirty = true;
    setPill('dirty');
    drawUsing(desired);
    applyDesired(true);
  });

  btnLinkAll.addEventListener('click', () => {
    desired.unlinkAll = false;
    desired.unlinks.clear();
    dirty = true;
    setPill('dirty');
    drawUsing(desired);
    applyDesired(true);
  });

  btnApply.addEventListener('click', () => applyDesired(true));

  async function poll(){
    try{
      const r = await fetch('/links/graph_state', {cache:'no-store'});
      const j = await r.json();

      const agents = (j.agents || j.labels || []).map(x => String(x));
      const unlinkAll = !!j.unlink_all;

      // IMPORTANT: backend returns "unlinks"; some variants use "unlink_pairs"
      const upairs = (j.unlink_pairs || j.unlinks || []);

      const set = new Set();
      for(const p of upairs){
        if(Array.isArray(p) && p.length===2){
          set.add(pairKey(p[0], p[1]));
        }
      }
      server = { unlinkAll, unlinks:set, agents };

      // Option 2: on first successful poll, copy server → desired and enable editing.
      // This prevents the "complete graph overwrite" problem.
      if(!initialized){
        syncDesiredFromServer();
        initialized = true;
      }

      // If not editing, keep desired in sync with server
      if(!dirty){
        syncDesiredFromServer();
      }


      // Always redraw (desired if editing, otherwise desired==server)
      drawUsing(dirty ? desired : server);

      // If editing, indicate drift if server differs (optional)
      if(dirty){
        // show warning if server has changed under us
        const drift = (server.unlinkAll !== desired.unlinkAll) || (server.unlinks.size !== desired.unlinks.size);
        if(drift) setPill('dirty');
      }

    }catch(e){
      statusEl.textContent = 'failed to load /links/graph_state';
      setPill('error');
    }
  }

  poll();
  setInterval(poll, 10000);
})();
</script>
</body>
</html>
"""



    return Response(html, mimetype="text/html")

@app.route("/config/save", methods=["POST"])
def config_save():
    data = request.get_json(force=True) if request.data else {}
    name = (data.get("name") or "").strip() or None
    try:
        path = _save_config_file(name)
        return jsonify({"ok": True, "file": os.path.basename(path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/config/load", methods=["POST"])
def config_load():
    data = request.get_json(force=True)
    filename = (data.get("filename") or "").strip()
    if not filename:
        return jsonify({"error":"filename required"}),400
    cfg_path = _resolve_config_path(filename, CURRENT_ENV)
    if not cfg_path: return jsonify({"error":"config not found"}),404
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        _load_config_data(obj)
        return jsonify({"ok": True, "file": os.path.basename(cfg_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/configs", methods=["GET"])
def list_configs():
    try:
        gdir = _global_configs_dir()
        ldir = _env_configs_dir(CURRENT_ENV)
        gl = sorted([f for f in os.listdir(gdir) if f.endswith(".json")]) if os.path.isdir(gdir) else []
        ll = sorted([f for f in os.listdir(ldir) if f.endswith(".json")]) if os.path.isdir(ldir) else []
        return jsonify({"global": gl, "room": ll})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------- Chat input handler ------------------- #


@app.route("/upload_txt", methods=["POST"])
def upload_txt():
    """Accept a .txt upload and return its decoded text (client appends into the next message)."""
    f = request.files.get("file")
    if f is None:
        return jsonify({"ok": False, "error": "missing file"}), 400
    filename = os.path.basename(f.filename or "upload.txt")
    if not filename.lower().endswith(".txt"):
        return jsonify({"ok": False, "error": "only .txt supported"}), 400
    raw = f.read() or b""
    if len(raw) > 512 * 1024:
        return jsonify({"ok": False, "error": "file too large"}), 400
    text = raw.decode("utf-8", errors="replace")
    return jsonify({"ok": True, "filename": filename, "text": text})

@app.route("/send", methods=["POST"])
def send():
    global STOPPED, ABORT_TOKEN, INVITE_QUEUE
    global UNLINK_ALL, UNLINK_PAIRS
    data=request.get_json(force=True)
    username=(data.get("username") or "you").strip() or "you"
    content=(data.get("content") or "").strip()
    if not content: return jsonify({"error":"empty"}),400

    lc = content.lower().strip()

    # --- routing commands: /links, /unlink, /link (no UI changes; command-only) ---
    if lc == "/links":
        system_line(_format_links_status())
        _save_config()
        return jsonify({"ok": True})

    m_all = re.match(r"^/(unlink|link)\s+all\s*$", content.strip(), re.I)
    if m_all:
        op = m_all.group(1).lower()
        if op == "unlink":
            UNLINK_ALL = True
            UNLINK_PAIRS = set()
            INVITE_QUEUE = deque()  # clear pending cross-bot invites
            _save_config()
            system_line("[links] Isolation ON: all bots are now isolated (each bot sees only you + itself).")
        else:
            UNLINK_ALL = False
            UNLINK_PAIRS = set()
            INVITE_QUEUE = deque()
            _save_config()
            system_line("[links] Full linking ON: all bots can see/talk to each other.")
        return jsonify({"ok": True})

    m_pair = re.match(r"^/(unlink|link)\s+(\S+)\s+(\S+)\s*$", content.strip(), re.I)
    if m_pair:
        op = m_pair.group(1).lower()
        a_tok = m_pair.group(2)
        b_tok = m_pair.group(3)

        a_key = _resolve_participant_key(a_tok)
        b_key = _resolve_participant_key(b_tok)

        if not a_key or not b_key or a_key == b_key:
            system_line("[links] Could not resolve both names. Use exact bot display names or their @keys.")
            return jsonify({"ok": True})

        # Pairwise operations imply we are not in global isolation mode
        if UNLINK_ALL:
            UNLINK_ALL = False

        pair = tuple(sorted((a_key, b_key)))

        if op == "unlink":
            UNLINK_PAIRS.add(pair)
            system_line(f"[links] Unlinked: {participants[a_key]['name']} ↔ {participants[b_key]['name']}")
        else:
            if pair in UNLINK_PAIRS:
                UNLINK_PAIRS.remove(pair)
            system_line(f"[links] Linked: {participants[a_key]['name']} ↔ {participants[b_key]['name']}")

        # prune invite queue: remove invites that violate new policy
        newq = deque()
        for item in list(INVITE_QUEUE):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                inviter, invitee = str(item[0]).lower(), str(item[1]).lower()
                if invitee in participants:
                    if inviter == USER_INVITER_KEY or inviter not in participants or _is_linked(inviter, invitee):
                        newq.append((inviter, invitee))
            else:
                # legacy: treat as user-invited
                invitee = str(item).lower()
                if invitee in participants:
                    newq.append((USER_INVITER_KEY, invitee))
        INVITE_QUEUE = newq

        _save_config()
        return jsonify({"ok": True})

    # --- existing commands (unchanged) ---
    if lc.startswith("/save "):  # keep legacy snapshot
        parts = content.split(maxsplit=1)
        name = parts[1].strip() if len(parts) > 1 else None
        try:
            path = _save_snapshot_file(name)
            system_line(f"[save] snapshot saved: {os.path.basename(path)}")
        except Exception as e:
            system_line(f"[save:error] {e}")
        return jsonify({"ok":True})

    if lc == "/save":  # allow just /save
        try:
            path = _save_snapshot_file(None)
            system_line(f"[save] snapshot saved: {os.path.basename(path)}")
        except Exception as e:
            system_line(f"[save:error] {e}")
        return jsonify({"ok":True})

    if lc.startswith("/load"):
        parts = content.split(maxsplit=1)
        if len(parts) < 2:
            system_line("[load] usage: /load <filename.json> — allowed only in a fresh room.")
            return jsonify({"ok": True})
        filename = parts[1].strip()
        snap_path = _resolve_snapshot_path(filename, CURRENT_ENV)
        if not snap_path:
            system_line("[load] file not found (check /_snapshots and env snapshots).")
            return jsonify({"ok": True})
        if not _can_load_into_current_room():
            system_line("[load] allowed only into a fresh room; use /reset first.")
            return jsonify({"ok": True})
        try:
            with open(snap_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            _load_snapshot_data(obj)
            system_line(f"[load] loaded {os.path.basename(snap_path)}")
        except Exception as e:
            system_line(f"[load:error] {e}")
        return jsonify({"ok": True})

    if lc == "/stop":
        STOPPED = True
        ABORT_TOKEN += 1
        remove_all_typing()
        _close_all_active()
        system_line("[control] generation stopped.")
        return jsonify({"ok": True})

    if lc == "/reset":
        return reset()

    if lc.startswith("/saveconfig"):
        parts = content.split(maxsplit=1)
        name = parts[1].strip() if len(parts) > 1 else None
        try:
            path = _save_config_file(name)
            system_line(f"[config] session config saved: {os.path.basename(path)}")
        except Exception as e:
            system_line(f"[config:error] {e}")
        return jsonify({"ok": True})

    if lc.startswith("/loadconfig"):
        parts = content.split(maxsplit=1)
        if len(parts) < 2:
            system_line("[config] usage: /loadconfig <filename.json> — starts a fresh chat.")
            return jsonify({"ok": True})
        cfg_path = _resolve_config_path(parts[1].strip(), CURRENT_ENV)
        if not cfg_path:
            system_line("[config] file not found (check /_configs and env configs).")
            return jsonify({"ok": True})
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            _load_config_data(obj)
            system_line(f"[config] loaded {os.path.basename(cfg_path)}")
        except Exception as e:
            system_line(f"[config:error] {e}")
        return jsonify({"ok": True})

    # Normal message
    user_line(username, content)

    # choose target + run turn
    mentions = parse_mentions(content)
    target_key = pick_target_for_turn(mentions)
    if not target_key:
        system_line("[notice] no agents yet; add one at right.")
        return jsonify({"ok": True})

    start_abort = ABORT_TOKEN
    invited = respond_once(target_key, start_abort)
    # chain limit logic
    depth = 1
    while invited and depth < CHAIN_HARD_LIMIT and ABORT_TOKEN == start_abort and not STOPPED:
        who = pick_target_for_turn(invited)
        if not who: break
        invited = respond_once(who, start_abort)
        depth += 1

    return jsonify({"ok": True})

if __name__ == "__main__":
    # boot default env
    cur = _read_current_env() or DEFAULT_ENV_NAME
    set_environment(cur)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","8080")), debug=False, threaded=True)
