"""PC Build Assistant — a local AI agent for GPU/CPU recommendations.

A Gradio chat interface backed by a LangChain agent with tools for hardware
spec lookup, bottleneck analysis, price checking, and value ranking. Prices
come from a manually maintained ``current_prices.csv``; the hardware
knowledge base is a ChromaDB vector store built from CSV files. The chat
model (Qwen3.6-35B-A3B) is served by a local llama-server; embeddings run
on Ollama.

Module map:
    config   — paths, constants, system prompt, examples
    styles   — the Gradio CSS
    data     — CSV loaders + vector store builder
    memory   — chat history + session context persistence
    llm      — shared model/embeddings/retriever singletons
    tools/   — the agent tools (hardware, bottleneck, pricing)
    agent    — agent cache + streaming chat handler
    ui       — Gradio Blocks builder
"""

__version__ = "1.0.0"