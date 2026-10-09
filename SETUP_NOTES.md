# Setup notes

How to run the assistant with **Qwen3.8-27B** on llama.cpp's `llama-server`,
split across two 16 GB GPUs (2× RTX 5070 Ti). Embeddings run on Ollama.

## 1. Update llama.cpp

The build in `C:\llama-cpp` is b8925 (April 2026). Qwen3.8 came out in
August 2026, so get a current release from
<https://github.com/ggml-org/llama.cpp/releases>. Use the Windows CUDA 13
x64 zip, plus the matching `cudart` zip if it's separate. Unzip both over
`C:\llama-cpp` and check:

```powershell
C:\llama-cpp\llama-server.exe --version
```

## 2. Download the model

Use a Q6_K quant, about 22.9 GB, which is close to lossless. Both
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
and [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF)
have one. Download only the model file. The `mmproj` file is the vision
encoder, which this app doesn't use.

```powershell
.venv\Scripts\hf download unsloth/Qwen3.8-27B-GGUF --include "*Q6_K.gguf" --local-dir C:\llama-cpp\models
```

Q5_K_M (~19 GB) leaves more room for context. Q8_0 (~29 GB) technically
fits in 32 GB but leaves almost nothing for the KV cache.

## 3. Start llama-server

```powershell
C:\llama-cpp\llama-server.exe `
  -m C:\llama-cpp\models\Qwen3.8-27B-Q6_K.gguf `
  --alias qwen3.8-27b --port 8033 `
  -ngl 99 -sm layer -ts 1,1 `
  -c 65536 -fa on -np 1 `
  --jinja --reasoning-format deepseek
```

Adjust the `-m` filename to whatever the download produced.

| Flag | Why |
|---|---|
| `--alias qwen3.8-27b` | Must match `CHAT_MODEL` in `config.py`. |
| `--port 8033` | Must match `LLAMA_SERVER_URL` in `config.py`. |
| `-ngl 99` | Every layer on the GPUs. No CPU or MoE offload is needed now. |
| `-sm layer -ts 1,1` | Half the layers on each card. |
| `-c 65536` | Context length. Qwen3.8 uses full attention in only 1 of every 4 layers, so its KV cache is small. |
| `-np 1` | One request slot, so all of the context goes to the single user. |
| `--jinja` | Required. Without the model's own chat template, tool calls and stop tokens break. |
| `--reasoning-format deepseek` | Returns Qwen's thinking in a separate `reasoning_content` field, so answers arrive clean. `ChatLlamaServer` in `llm.py` sends that reasoning back between tool calls, which is how Qwen3.8 was trained to work. |

**Don't let VRAM overflow.** On Windows, the NVIDIA driver quietly spills
into system RAM when VRAM runs out, and generation slows to a crawl.
Watch `nvidia-smi` after loading and keep about 1 GB free per card. Ollama's
embedding model also takes a little VRAM. If memory is tight, lower `-c` to
`32768` or add `-ctk q8_0 -ctv q8_0` to halve the KV cache.

**Split modes.** `layer` pipelines the two cards, so one token runs at about
single-card speed, and the second card mostly adds memory. `row` and
`tensor` (experimental) run both cards on each layer at once, but on
consumer PCIe that can be faster or slower. Measure it:

```powershell
C:\llama-cpp\llama-bench.exe -m C:\llama-cpp\models\Qwen3.8-27B-Q6_K.gguf -ngl 99 -fa 1 -sm layer,row
```

## 4. Run the app

1. Start `llama-server` and wait for `server is listening`.
2. Make sure Ollama is running and has `nomic-embed-text`
   (`ollama pull nomic-embed-text`).
3. Run `python app.py` and open the printed URL.

The first message starts the MCP hardware server as a subprocess, which
takes about 2 seconds. Later messages reuse it, and it exits with the app.

**Thinking.** `THINKING` in `config.py` switches Qwen3.8's reasoning on or
off. It's sent with each request, so the server doesn't need a restart.
Each mode uses the sampling settings from the model card (`SAMPLING` in
`config.py`). Thinking off is much faster per turn.

## 5. Evaluate

`evals/run_eval.py` puts a fixed set of questions through the agent. For
each one it checks the tools called, their arguments, and the final answer.

```powershell
python -m evals.run_eval              # config defaults
python -m evals.run_eval --no-think   # compare against thinking off
python -m evals.run_eval --repeat 3   # find flaky cases
python -m evals.run_eval --list       # show the cases
```

Results go to `evals/results/`. Run the eval after changing the model, the
system prompt, or a tool. If a case fails with "tool-call/thinking markup
leaked into the answer", the server isn't parsing Qwen3.8's tool calls or
thinking. Check `--jinja` and `--reasoning-format`, and that llama.cpp is
current.

## 6. Use the MCP server from other clients

The tools and datasets live in an MCP server
(`pc_build_assistant/mcp_server.py`). The chat app is one client; any MCP
client can launch the same server over stdio. It needs Ollama running for
semantic search, and it doesn't need llama-server.

It offers:

- **Tools:** `search_hardware_database`, `compare_components`,
  `check_cpu_gpu_bottleneck`, `check_current_value`, `find_best_value`
- **Resources:** `hardware://gpus`, `hardware://cpus`, `hardware://prices`
  (the raw CSVs)

**Claude Desktop.** Add this to `%APPDATA%\Claude\claude_desktop_config.json`,
or open it from Settings → Developer → Edit Config, then restart Claude
Desktop:

```json
{
  "mcpServers": {
    "pc-hardware": {
      "command": "D:\\Desktop\\Projects\\Personal\\PCAssistant\\.venv\\Scripts\\python.exe",
      "args": ["-m", "pc_build_assistant.mcp_server"],
      "env": { "PYTHONPATH": "D:\\Desktop\\Projects\\Personal\\PCAssistant" }
    }
  }
}
```

**Claude Code:**

```powershell
claude mcp add pc-hardware -e PYTHONPATH=D:\Desktop\Projects\Personal\PCAssistant -- D:\Desktop\Projects\Personal\PCAssistant\.venv\Scripts\python.exe -m pc_build_assistant.mcp_server
```

**MCP Inspector**, for poking at the tools by hand (needs Node.js):

```powershell
npx @modelcontextprotocol/inspector .venv\Scripts\python.exe -m pc_build_assistant.mcp_server
```

stdout carries the MCP protocol, so any logging added to the tools must
go to stderr.

## Going back to Qwen3.6-35B-A3B

`C:\llama-cpp\models\Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` (26.6 GB) also fits
entirely on the two cards with the same command. Change `-m` and `--alias`,
and lower `-c` if VRAM runs short. Set `CHAT_MODEL` to match. The `SAMPLING`
presets in `config.py` are tuned for Qwen3.8, so check them against the
Qwen3.6 model card. It's a mixture-of-experts model with about 3B active
parameters, so it's much faster per token.
