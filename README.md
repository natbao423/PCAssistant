# PCAssistant

A local AI agent that answers PC hardware questions — GPU and CPU specs,
value comparisons, and CPU-GPU bottleneck analysis — without sending
anything to a cloud API. It runs a quantized Qwen model on consumer
hardware and answers from a curated spec database rather than from model
memory, so it says "not in the database" instead of inventing plausible-
sounding numbers.

Built with LangChain and Gradio. The agent has five tools covering spec
lookup, side-by-side comparison, bottleneck analysis, price/value
checking, and budget-constrained ranking. Retrieval combines ChromaDB
semantic search with exact model-name matching, and a session context
tracker carries facts like the user's budget across turns so older
messages can be dropped from the context window without losing
continuity. The chat backend is switchable between Ollama and
llama.cpp's llama-server, the latter tuned with MoE layer offloading to
fit a 35B mixture-of-experts model on a 16 GB GPU.
