# PCAssistant

A local AI agent that answers PC hardware questions — GPU and CPU specs,
value comparisons, and CPU-GPU bottleneck analysis — without sending
anything to a cloud API. It runs a quantized Qwen3.8-27B on consumer
GPUs and answers from a curated spec database rather than from model
memory, so it says "not in the database" instead of inventing plausible-
sounding numbers.

Built with LangChain, Gradio, and the Model Context Protocol (MCP). The
hardware data and five tools — spec lookup, side-by-side comparison,
bottleneck scoring, price/value checking, and budget-constrained ranking —
live in an MCP server, which also exposes the GPU, CPU, and price datasets
as resources. The chat agent is an MCP client of that server, and so can
any other MCP client, such as Claude Desktop. Retrieval combines ChromaDB
semantic search with exact model-name matching, and a session context
tracker carries facts like the user's budget across turns so older
messages can be dropped from the context window without losing
continuity. The chat backend is switchable between Ollama and
llama.cpp's llama-server, the latter serving Qwen3.8-27B split across
two 16 GB GPUs.

## Running

See [SETUP_NOTES.md](SETUP_NOTES.md) for the llama-server command, then:

```
python app.py                  # the chat UI (starts the MCP server itself)
python -m pytest tests         # name-matching unit tests (no model needed)
python -m evals.run_eval       # end-to-end eval against the live model
```

```
                    MCP (stdio)
 Gradio UI ─ agent ─────────────┐
                                ├─► pc_build_assistant.mcp_server
 Claude Desktop / Claude Code ──┘     5 tools · 3 dataset resources
                                      CSVs + ChromaDB (Ollama embeddings)
```
