"""The hardware tools.

Re-exports the five tool callables plus ``ALL_TOOLS``, the list the MCP
server (``mcp_server``) registers. The agent never imports these directly;
it reaches them through MCP.
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

__all__ = [
    "search_hardware_database",
    "compare_components",
    "check_cpu_gpu_bottleneck",
    "check_current_value",
    "find_best_value",
    "ALL_TOOLS",
]