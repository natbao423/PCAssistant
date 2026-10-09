"""Configuration constants, paths, and the agent system prompt.

All filesystem paths resolve relative to the project root (the directory
containing ``app.py``), so the existing data files (``all_gpus.csv``,
``cpus.csv``, ``current_prices.csv``, ``chroma_db/``) keep working without
being moved.
"""

from pathlib import Path

# Project root = one level up from this package directory.
BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = str(BASE_DIR / "chroma_db")
INDEX_RECORD = Path(CHROMA_PATH) / "indexed_files.json"
HISTORY_FILE = BASE_DIR / "chat_history.json"
CONTEXT_FILE = BASE_DIR / "session_context.json"
PRICE_PATH = BASE_DIR / "current_prices.csv"

INDEX_VERSION = 3

CSV_SOURCES: list[tuple[str, str]] = [
    ("all_gpus.csv", "gpu_specs_and_features"),
    ("cpus.csv", "cpu_specs"),
]

# Minimum CPU gaming-score thresholds keyed by GPU performance floor.
# Each entry is ``(gpu_score_floor, min_cpu_score, tier_label)``.
BOTTLENECK_TIERS: list[tuple[float, float, str]] = [
    (90, 91.0, "flagship (5090-class)"),
    (65, 87.0, "high-end (5080/4090-class)"),
    (50, 82.0, "upper mid-range (5070 Ti/4080S/9070 XT-class)"),
    (40, 75.0, "mid-range (5070/4070TiS/9070-class)"),
    (25, 66.0, "entry mid-range (5060 Ti/4060 Ti/9060 XT-class)"),
    (0,  60.0, "budget (4060/5060-class)"),
]

# Which backend serves the chat model.
#   "llama_server"— llama.cpp llama-server running Qwen3.8-27B across both
#                   GPUs. See SETUP_NOTES.md for the startup command.
#   "ollama"      — simple, uses a model you `ollama pull`. Good for a
#                   smoke test or when llama.cpp isn't set up yet.
CHAT_BACKEND = "llama_server"

# --- llama-server backend (used when CHAT_BACKEND == "llama_server") ---
# OpenAI-compatible endpoint. Port must match llama-server's --port.
LLAMA_SERVER_URL = "http://localhost:8033/v1"
# Must match llama-server's --alias.
CHAT_MODEL = "qwen3.8-27b"

# Qwen3.8 thinks before answering by default. Thinking gives better tool
# choices on harder questions; turning it off is much faster per turn.
# Sent per request, so it can be flipped without restarting the server.
THINKING = True

# Qwen3.8-27B model-card sampling presets, keyed by THINKING.
SAMPLING: dict[bool, dict[str, float]] = {
    True:  {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 0.0},
    False: {"temperature": 0.7, "top_p": 0.80, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5},
}

# Per-response token cap. Thinking can run long before the answer starts.
MAX_TOKENS: dict[bool, int] = {True: 8192, False: 2048}

# --- Ollama backend (used when CHAT_BACKEND == "ollama") ---
# Must match a model shown by `ollama list`. Pull it first, e.g.
#   ollama pull qwen3.5:9b-q8_0
OLLAMA_CHAT_MODEL = "qwen3.5:9b-q8_0"

# Embeddings always run on Ollama (tiny model). Pull once:
#   ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"

# Cap on prior chat messages sent to the model per turn. Keeps prompts
# (and so prompt-processing time) short; session context preserves key
# facts like the budget across truncation.
MAX_HISTORY_MESSAGES = 12

SYSTEM_PROMPT = (
    "You are a PC hardware expert. You MUST call a tool before every response. "
    "Never answer from memory.\n\n"
    "TOOL ROUTING:\n"
    "- Compare two components → compare_components (ALWAYS use this for vs/compare queries)\n"
    "- Specs/features of a single component → search_hardware_database\n"
    "- Price or value check → check_current_value\n"
    "- Budget recommendation → find_best_value\n"
    "- CPU+GPU pairing → check_cpu_gpu_bottleneck\n\n"
    "RESPONSE FORMAT:\n"
    "- Give a direct answer. Pick a winner and explain why.\n"
    "- If a tool says NOT FOUND, tell the user — NEVER fill in specs from memory.\n"
    "- If data is missing, say so — do not guess.\n"
    "- Do NOT list numbered options like '1. Build a PC? 2. Upgrade?'\n"
    "- Do NOT output raw JSON.\n"
)

EXAMPLES: list[str] = [
    "Compare the RTX 5080 and RX 9070 XT for 4K gaming",
    "What's the best GPU under $600?",
    "Will a Ryzen 5 5600X bottleneck an RTX 5080?",
    "Ryzen 7 9800X3D vs Core i7-14700K for gaming",
    "Best CPU-GPU combo under $1000 total?",
    "Does the RTX 5070 support DLSS 4.5 Multi Frame Generation?",
]