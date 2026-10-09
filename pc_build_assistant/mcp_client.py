"""MCP client side: connects the agent to the hardware tool server.

The agent gets its tools here, over MCP, rather than importing them: the
server (``mcp_server``) runs as a subprocess speaking stdio, and its tools
arrive as LangChain tools via ``langchain-mcp-adapters``.
"""

import asyncio
import sys

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .config import BASE_DIR

SERVER_NAME = "pc-hardware"


def server_connection() -> dict:
    """stdio connection settings that launch the hardware MCP server."""
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "pc_build_assistant.mcp_server"],
        "cwd": str(BASE_DIR),
    }


def _with_text_content(tool: BaseTool) -> BaseTool:
    """Make an MCP tool return plain text instead of content blocks.

    The adapters return results as a list of content blocks, even for a
    single text result. The tools only ever return text, so join it back
    into a string: the model then sees exactly what the Python functions
    return, on every backend.
    """
    call = tool.coroutine

    async def text_only(**kwargs):
        content, artifact = await call(**kwargs)
        if isinstance(content, list) and all(
            isinstance(b, dict) and b.get("type") == "text" for b in content
        ):
            content = "\n".join(b["text"] for b in content)
        return content, artifact

    tool.coroutine = text_only
    return tool


class McpToolbox:
    """One long-lived MCP session to the hardware server.

    The adapters' default opens a new session for every tool call, which
    over stdio means a new server process reloading the vector store each
    time. This keeps a single session open instead. It lives in its own
    background task because the MCP SDK's anyio task groups must be entered
    and exited by the same task, while tool calls come from many tasks.

    The session belongs to the event loop that started it; a call from a
    different loop (or after the server exits) starts a fresh one.
    """

    def __init__(self) -> None:
        self._client = MultiServerMCPClient({SERVER_NAME: server_connection()})
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._tools: list[BaseTool] = []
        self._error: Exception | None = None

    async def get_tools(self) -> list[BaseTool]:
        """Return the server's tools, starting the session if needed.

        Raises:
            RuntimeError: If the server could not be started.
        """
        loop = asyncio.get_running_loop()
        if self._task is None or self._task.done() or self._loop is not loop:
            # No await between the check and create_task, so concurrent
            # callers on this loop can't start two sessions.
            self._loop = loop
            self._ready, self._stop = asyncio.Event(), asyncio.Event()
            self._tools, self._error = [], None
            self._task = loop.create_task(self._serve())
        await self._ready.wait()
        if self._error is not None:
            raise RuntimeError(
                f"Couldn't start the MCP hardware server: {self._error}"
            ) from self._error
        return self._tools

    async def _serve(self) -> None:
        try:
            async with self._client.session(SERVER_NAME) as session:
                self._tools = [
                    _with_text_content(tool)
                    for tool in await load_mcp_tools(
                        session, server_name=SERVER_NAME,
                    )
                ]
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            self._error = exc
        finally:
            self._ready.set()

    async def close(self) -> None:
        """Close the session and let the server process exit."""
        if self._task is not None and not self._task.done():
            self._stop.set()
            await self._task


toolbox = McpToolbox()
