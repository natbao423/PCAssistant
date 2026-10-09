"""Run the eval cases against a live chat model and grade the results.

Each case sends one question to the agent (same tools and system prompt as
the app, without session context or chat history), then checks which
tools were called with which arguments, and what the final answer says.
Tools are reached through the MCP hardware server, as in the app.

Usage (from the project root, with Ollama running for embeddings and the
chat backend up):

    python -m evals.run_eval                     # backend/model from config
    python -m evals.run_eval --no-think          # Qwen3.8 thinking off
    python -m evals.run_eval --repeat 3          # catch flaky cases
    python -m evals.run_eval --only price_9950x,unknown_gpu -v
    python -m evals.run_eval --backend ollama --model qwen3.5:9b-q8_0

Results are saved as JSON under evals/results/. Exit code is 1 if any run
failed, so it can gate changes to prompts or tools.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from pc_build_assistant import config
from pc_build_assistant.agent import build_agent, strip_thinking
from pc_build_assistant.data import known_models
from pc_build_assistant.llm import build_llm
from pc_build_assistant.mcp_client import toolbox

from .cases import CASES, Case, Pattern

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Graph steps per question (each tool round is two: model, then tools).
RECURSION_LIMIT = 25

# Signs of a tool call or JSON leaking into the answer text.
_LEAK_RE = re.compile(r"<tool_call>|</?think>|\{\s*\"name\"\s*:", re.I)


# ── Grading ─────────────────────────────────────────────────────────────

def _resolve(pattern: Pattern) -> str:
    return pattern() if callable(pattern) else pattern


def _arg_matches(actual: object, expected: object) -> bool:
    """Numbers within 1%, strings case-insensitively, else equality."""
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - expected) <= abs(expected) * 0.01
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str):
        return str(actual).strip().casefold() == expected.casefold()
    return actual == expected


def grade(case: Case, calls: list[dict], answer: str, truncated: bool) -> list[str]:
    """Return the reasons a run failed (empty list = pass).

    Args:
        case: The case being graded.
        calls: Tool calls made, as ``{"name": ..., "args": {...}}``.
        answer: Final answer with thinking stripped.
        truncated: Whether any model response hit the token cap.
    """
    failures: list[str] = []

    if not calls:
        failures.append("no tool was called")
    for name, want in case.calls:
        if not any(
            c["name"] == name
            and all(_arg_matches(c["args"].get(k), v) for k, v in want.items())
            for c in calls
        ):
            failures.append(f"missing call {name}({_fmt_args(want)})")

    if case.models:
        index = known_models()
        named: set[str] = set()
        for c in calls:
            for value in c["args"].values():
                for mention in index.find_all(str(value)):
                    named.update(mention.models)
        missing = [m for m in case.models if m not in named]
        if missing:
            failures.append(f"tool args never named {', '.join(missing)}")

    if len(calls) > case.max_calls:
        failures.append(f"{len(calls)} tool calls (max {case.max_calls})")
    if case.check:
        message = case.check(calls, answer)
        if message:
            failures.append(message)

    if truncated:
        failures.append("hit the max_tokens cap")
    if not answer:
        failures.append("empty answer")
        return failures
    if _LEAK_RE.search(answer):
        failures.append("tool-call/thinking markup leaked into the answer")
    for p in case.include:
        pattern = _resolve(p)
        if not re.search(pattern, answer, re.I | re.S):
            failures.append(f"answer lacks /{pattern}/")
    for p in case.exclude:
        pattern = _resolve(p)
        if re.search(pattern, answer, re.I | re.S):
            failures.append(f"answer contains /{pattern}/")
    return failures


# ── Running ─────────────────────────────────────────────────────────────

def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


async def run_case(agent, case: Case) -> dict:
    """Run one case once and return a result record."""
    start = time.perf_counter()
    state = {"messages": []}
    error = None
    try:
        # Stream full states rather than invoke(), so a run that errors
        # part-way still reports the tool calls it made.
        async for state in agent.astream(
            {"messages": [{"role": "user", "content": case.prompt}]},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="values",
        ):
            pass
    except Exception as exc:  # connection errors, runaway tool loops, ...
        error = f"error: {type(exc).__name__}: {str(exc).splitlines()[0]}"
    seconds = time.perf_counter() - start

    calls: list[dict] = []
    output_tokens = 0
    truncated = False
    for msg in state["messages"]:
        if getattr(msg, "type", "") != "ai":
            continue
        calls += [
            {"name": c["name"], "args": c.get("args") or {}}
            for c in msg.tool_calls
        ]
        output_tokens += (msg.usage_metadata or {}).get("output_tokens", 0)
        meta = msg.response_metadata or {}
        if (meta.get("finish_reason") or meta.get("done_reason")) == "length":
            truncated = True

    final = state["messages"][-1] if error is None else None
    answer = strip_thinking(final.content) if final and isinstance(final.content, str) else ""
    failures = [error] if error else grade(case, calls, answer, truncated)
    return {
        "id": case.id, "passed": not failures, "seconds": seconds,
        "calls": calls, "answer": answer, "output_tokens": output_tokens,
        "failures": failures,
    }


def check_backend(backend: str) -> str | None:
    """Return an error message if the chat backend isn't reachable."""
    url = (
        config.LLAMA_SERVER_URL.rstrip("/") + "/models"
        if backend == "llama_server"
        else "http://localhost:11434/api/tags"
    )
    try:
        urllib.request.urlopen(url, timeout=5).close()
    except OSError as exc:
        return f"Can't reach the {backend} backend at {url} ({exc})."
    return None


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["llama_server", "ollama"],
                        default=config.CHAT_BACKEND)
    parser.add_argument("--model", help="model name (default: from config)")
    think = parser.add_mutually_exclusive_group()
    think.add_argument("--think", dest="thinking", action="store_true",
                       default=None, help="force thinking on")
    think.add_argument("--no-think", dest="thinking", action="store_false",
                       help="force thinking off")
    parser.add_argument("--repeat", type=int, default=1,
                        help="runs per case (models are not deterministic)")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print every answer")
    parser.add_argument("--no-save", action="store_true",
                        help="don't write a results JSON")
    args = parser.parse_args()

    cases = CASES
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {c.id for c in CASES}
        if unknown:
            parser.error(f"unknown case ids: {', '.join(sorted(unknown))}")
        cases = [c for c in CASES if c.id in wanted]
    if args.list:
        for c in cases:
            print(f"{c.id:24} {c.prompt}")
        return 0

    error = check_backend(args.backend)
    if error:
        print(error, "Start it first (see SETUP_NOTES.md).")
        return 2

    thinking = config.THINKING if args.thinking is None else args.thinking
    model_name = args.model or (
        config.CHAT_MODEL if args.backend == "llama_server"
        else config.OLLAMA_CHAT_MODEL
    )
    try:
        # Tools come from the MCP server, exactly as in the app.
        tools = await toolbox.get_tools()
    except RuntimeError as exc:
        print(exc)
        return 2
    agent = build_agent(
        tools, model=build_llm(args.backend, args.model, thinking),
    )
    label = f"{model_name} ({args.backend}, thinking {'on' if thinking else 'off'})"
    print(f"\nEvaluating {label}: {len(cases)} cases x {args.repeat}, "
          f"{len(tools)} tools over MCP\n")

    results: list[dict] = []
    try:
        for case in cases:
            for _ in range(args.repeat):
                r = await run_case(agent, case)
                results.append(r)
                calls = "; ".join(
                    f"{c['name']}({_fmt_args(c['args'])})" for c in r["calls"]
                )
                print(f"{'PASS' if r['passed'] else 'FAIL'}  {case.id:24} "
                      f"{r['seconds']:6.1f}s  {calls or '(no tool calls)'}")
                for failure in r["failures"]:
                    print(f"      - {failure}")
                if args.verbose or not r["passed"]:
                    snippet = r["answer"] if args.verbose else r["answer"][:300]
                    print("      answer: " + snippet.replace("\n", "\n              "))
    finally:
        await toolbox.close()

    # ── Summary ─────────────────────────────────────────────────────
    passed = sum(r["passed"] for r in results)
    seconds = [r["seconds"] for r in results]
    tokens = sum(r["output_tokens"] for r in results)
    print(f"\n{passed}/{len(results)} runs passed — {label}")
    print(f"Latency: median {statistics.median(seconds):.1f}s, "
          f"max {max(seconds):.1f}s, total {sum(seconds):.0f}s")
    if tokens:
        print(f"Output: {tokens} tokens, ~{tokens / sum(seconds):.0f} tok/s "
              f"(includes prompt processing and tool time)")
    if args.repeat > 1:
        for case in cases:
            runs = [r for r in results if r["id"] == case.id]
            ok = sum(r["passed"] for r in runs)
            if 0 < ok < len(runs):
                print(f"Flaky: {case.id} passed {ok}/{len(runs)}")

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_model = re.sub(r"[^\w.-]+", "_", model_name)
        path = RESULTS_DIR / f"{stamp}_{safe_model}_{'think' if thinking else 'nothink'}.json"
        path.write_text(json.dumps({
            "model": model_name, "backend": args.backend, "thinking": thinking,
            "repeat": args.repeat, "passed": passed, "total": len(results),
            "results": results,
        }, indent=2, default=str), encoding="utf-8")
        print(f"Saved {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
