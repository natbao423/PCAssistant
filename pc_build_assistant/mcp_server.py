"""MCP server for the hardware database and scoring tools.

Exposes the five agent tools (spec search, comparison, bottleneck scoring,
price/value checks, budget ranking) and the three source datasets over the
Model Context Protocol. The chat app's agent starts this as a subprocess
and reaches every tool through it; any other MCP client (Claude Desktop,
Claude Code, ...) can use it the same way.

Run over stdio (what MCP clients launch):

    python -m pc_build_assistant.mcp_server

Needs Ollama running with ``nomic-embed-text`` for semantic search.
stdout carries the protocol, so all logging must go to stderr.
"""

import contextlib
import sys

from mcp.server.fastmcp import FastMCP

from .config import BASE_DIR, PRICE_PATH
from .retrieval import get_vectorstore
from .tools import ALL_TOOLS

mcp = FastMCP(
    "pc-hardware",
    log_level="WARNING",  # INFO logs every request (to stderr)
    instructions=(
        "A curated database of desktop GPUs and CPUs: specs, features, "
        "benchmark scores, MSRP and street prices. Answer hardware questions "
        "from these tools, not from memory. If a tool says NOT FOUND, the "
        "part is not in the database: say so rather than supplying specs. "
        "Use compare_components for any 'X vs Y' question, "
        "check_cpu_gpu_bottleneck for pairings, find_best_value for budget "
        "picks, check_current_value for one part's price, and "
        "search_hardware_database for specs and features."
    ),
)

for tool in ALL_TOOLS:
    mcp.add_tool(tool)


# ── Datasets as resources ───────────────────────────────────────────────

@mcp.resource(
    "hardware://gpus", mime_type="text/csv",
    description="GPU specs, features, MSRP and relative 4K performance.",
)
def gpu_dataset() -> str:
    return (BASE_DIR / "all_gpus.csv").read_text(encoding="utf-8")


@mcp.resource(
    "hardware://cpus", mime_type="text/csv",
    description="CPU specs, MSRP and relative 1080p gaming performance.",
)
def cpu_dataset() -> str:
    return (BASE_DIR / "cpus.csv").read_text(encoding="utf-8")


@mcp.resource(
    "hardware://prices", mime_type="text/csv",
    description="Street prices by model, with retailer and date checked.",
)
def price_dataset() -> str:
    return PRICE_PATH.read_text(encoding="utf-8")


def main() -> None:
    """Load the vector store, then serve MCP over stdio."""
    # Load up front so the first search isn't slow and a missing Ollama
    # fails at startup. Anything printed meanwhile goes to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        get_vectorstore()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
