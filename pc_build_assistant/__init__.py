"""PC Build Assistant — a local AI agent for GPU/CPU recommendations.

A Gradio chat interface backed by a LangChain agent with tools for hardware
spec lookup, bottleneck analysis, price checking, and value ranking. Prices
come from a manually maintained ``current_prices.csv``; the hardware
knowledge base is a ChromaDB vector store built from CSV files. The chat
model (Qwen3.8-27B) is served by a local llama-server; embeddings run on
Ollama.

Module map:
    config     — paths, constants, system prompt, examples
    styles     — the Gradio CSS
    data       — CSV loaders + vector store builder
    matching   — model-name matching between text and the CSVs
    retrieval  — embeddings, vector store, retriever (lazy)
    tools/     — the hardware tools (hardware, bottleneck, pricing)
    mcp_server — MCP server exposing the tools and datasets
    mcp_client — the agent's long-lived MCP session to that server
    memory     — chat history + session context persistence
    llm        — the chat model
    agent      — agent cache + streaming chat handler
    ui         — Gradio Blocks builder
"""

__version__ = "1.0.0"