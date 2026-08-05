"""Agent tools and their UI status messages.

Re-exports the five tool callables plus ``ALL_TOOLS`` (the list passed to
``create_agent``) and ``TOOL_STATUS`` (the per-tool "now running" messages
shown in the chat stream).
"""

from .bottleneck import check_cpu_gpu_bottleneck
from .hardware import search_hardware_database
from .pricing import (
    check_current_value,
    compare_components,
    find_best_value,
)

ALL_TOOLS = [
    search_hardware_database,
    compare_components,
    check_cpu_gpu_bottleneck,
    check_current_value,
    find_best_value,
]

# Friendly status shown while each tool runs. Keys must match tool names.
TOOL_STATUS: dict[str, str] = {
    "search_hardware_database": "🔍 Searching hardware database…",
    "compare_components":        "⚖️ Comparing components side-by-side…",
    "check_cpu_gpu_bottleneck":  "⚙️ Analysing CPU–GPU bottleneck…",
    "check_current_value":       "💰 Checking current prices…",
    "find_best_value":           "📊 Ranking components by value…",
}
STATUS_FALLBACK = "🔧 Running tool…"

__all__ = [
    "search_hardware_database",
    "compare_components",
    "check_cpu_gpu_bottleneck",
    "check_current_value",
    "find_best_value",
    "ALL_TOOLS",
    "TOOL_STATUS",
    "STATUS_FALLBACK",
]