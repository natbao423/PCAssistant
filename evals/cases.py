"""Eval cases: a question, the tool calls it should produce, and checks on
the final answer.

Expected facts (prices, top picks) are computed from the tools at grading
time rather than hard-coded, so the cases stay valid when the CSVs are
updated. Answer patterns are case-insensitive regexes.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from pc_build_assistant.matching import tokenize
from pc_build_assistant.tools import check_current_value, find_best_value

# A pattern is a regex string, or a callable returning one (evaluated when
# grading, so it reflects the current CSVs).
Pattern = str | Callable[[], str]

# How a model says a component isn't in the database.
NOT_FOUND = (
    r"(\bnot|n't)\b[^.\n]{0,25}?\b(in (the|our|my|this)\b|found|listed"
    r"|available|included|appear|exist|recogni[sz]ed|have|contain)"
    r"|(couldn't|could not|can't|cannot|unable to) (find|locate)"
    r"|\bno (data|information|record|entry|listing|match)"
)


@dataclass
class Case:
    """One eval question.

    Attributes:
        id: Short unique name, used with ``--only``.
        prompt: The user message.
        calls: Required tool calls as ``(tool_name, args_subset)``. Each
            must be matched by some call with that name whose args include
            those values (numbers within 1%, strings case-insensitively).
        models: Canonical model names that must appear in tool arguments.
        include: Patterns the final answer must contain.
        exclude: Patterns the final answer must not contain.
        check: Optional extra check given ``(calls, answer)``; returns a
            failure message, or None if the run passes it.
        max_calls: Fail if the agent makes more tool calls than this
            (catches runaway exploration that makes answers slow).
    """

    id: str
    prompt: str
    calls: list[tuple[str, dict]] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    include: list[Pattern] = field(default_factory=list)
    exclude: list[Pattern] = field(default_factory=list)
    check: Callable[[list[dict], str], str | None] | None = None
    max_calls: int = 6


# ── Expected-value helpers ──────────────────────────────────────────────

def mentions(model: str) -> str:
    """Regex for a model name as an answer might write it.

    Matches from the model number on, with flexible spacing, so "Ryzen 9
    9950X" accepts "9950X" and "9950 X" but not "9950X3D".
    """
    tokens = list(tokenize(model))
    start = next(
        (i for i, t in enumerate(tokens) if t.isdigit() and len(t) >= 3), 0,
    )
    return r"\b" + r"[\s-]*".join(map(re.escape, tokens[start:])) + r"\b"


def top_pick(component_type: str, budget: float) -> Callable[[], str]:
    """Pattern for the #1 result of find_best_value."""
    def pattern() -> str:
        out = find_best_value(budget, component_type)
        return mentions(re.search(r"^1\. (.+?) — ", out, re.M).group(1))
    return pattern


def price_of(model: str) -> Callable[[], str]:
    """Pattern for the price check_current_value reports, e.g. "$515"."""
    def pattern() -> str:
        out = check_current_value(model)
        dollars = re.search(r"Current price: \$(\d+)", out).group(1)
        return rf"\$\s?{dollars}\b"
    return pattern


def budget_split_within(total: float) -> Callable[[list[dict], str], str | None]:
    """Check that some GPU + CPU find_best_value budgets fit in ``total``."""
    def check(calls: list[dict], answer: str) -> str | None:
        budgets = {"GPU": [], "CPU": []}
        for c in calls:
            kind = str(c["args"].get("component_type", "")).upper()
            if c["name"] == "find_best_value" and kind in budgets:
                try:
                    budgets[kind].append(float(c["args"]["budget"]))
                except (KeyError, TypeError, ValueError):
                    pass
        if not budgets["GPU"] or not budgets["CPU"]:
            return None  # the missing call is reported separately
        best = min(budgets["GPU"]) + min(budgets["CPU"])
        if best > total * 1.01:
            return f"GPU + CPU budgets add up to ${best:.0f}, over ${total:.0f}"
        return None
    return check


# ── Cases ───────────────────────────────────────────────────────────────

CASES: list[Case] = [
    # The examples shown in the UI.
    Case(
        id="compare_gpus",
        prompt="Compare the RTX 5080 and RX 9070 XT for 4K gaming",
        calls=[("compare_components", {})],
        models=["RTX 5080", "RX 9070 XT"],
        include=[mentions("RTX 5080"), mentions("RX 9070 XT")],
    ),
    Case(
        id="best_gpu_600",
        prompt="What's the best GPU under $600?",
        calls=[("find_best_value", {"budget": 600, "component_type": "GPU"})],
        include=[top_pick("GPU", 600)],
    ),
    Case(
        id="bottleneck_5600x_5080",
        prompt="Will a Ryzen 5 5600X bottleneck an RTX 5080?",
        calls=[("check_cpu_gpu_bottleneck", {})],
        models=["Ryzen 5 5600X", "RTX 5080"],
        include=[r"bottleneck"],
    ),
    Case(
        id="compare_cpus",
        prompt="Ryzen 7 9800X3D vs Core i7-14700K for gaming",
        calls=[("compare_components", {})],
        models=["Ryzen 7 9800X3D", "Core i7-14700K"],
        include=[mentions("Ryzen 7 9800X3D"), mentions("Core i7-14700K")],
    ),
    Case(
        id="combo_1000",
        prompt="Best CPU-GPU combo under $1000 total?",
        calls=[
            ("find_best_value", {"component_type": "GPU"}),
            ("find_best_value", {"component_type": "CPU"}),
        ],
        check=budget_split_within(1000),
        max_calls=8,
    ),
    Case(
        id="dlss_5070",
        prompt="Does the RTX 5070 support DLSS 4.5 Multi Frame Generation?",
        calls=[("search_hardware_database", {})],
        models=["RTX 5070"],
        include=[r"\byes\b|supports?", r"6x|multi frame"],
        exclude=[r"(does not|doesn't|not) support"],
    ),

    # Name resolution: the right variant, with the right price.
    Case(
        id="price_9950x",
        prompt="How much is the Ryzen 9 9950X right now?",
        calls=[("check_current_value", {})],
        models=["Ryzen 9 9950X"],
        include=[price_of("Ryzen 9 9950X")],
        exclude=[price_of("Ryzen 9 9950X3D")],
    ),
    Case(
        id="price_7700_msrp_only",
        prompt="What's the current price of the Ryzen 7 7700?",
        calls=[("check_current_value", {})],
        models=["Ryzen 7 7700"],
        include=[price_of("Ryzen 7 7700")],
        exclude=[price_of("Ryzen 7 7700X")],
    ),
    Case(
        id="sloppy_names",
        prompt="5070ti or 9070xt, which is the better value?",
        calls=[("compare_components", {})],
        models=["RTX 5070 Ti", "RX 9070 XT"],
    ),
    Case(
        id="cpu_budget_300",
        prompt="Best gaming CPU under $300?",
        calls=[("find_best_value", {"budget": 300, "component_type": "CPU"})],
        include=[top_pick("CPU", 300)],
    ),
    Case(
        id="pairing_12400f_9070xt",
        prompt="Is a Core i5-12400F enough for an RX 9070 XT?",
        calls=[("check_cpu_gpu_bottleneck", {})],
        models=["Core i5-12400F", "RX 9070 XT"],
    ),

    # Grounding: parts that don't exist must not get invented specs.
    Case(
        id="unknown_gpu",
        prompt="What are the specs of the RTX 6090?",
        include=[NOT_FOUND],
        # Specs pinned on the 6090 itself. Real specs of other cards from
        # the database (offered as alternatives) are fine.
        exclude=[r"6090[^.\n]{0,60}?(\d[\d,]*\s*CUDA|\d+\s*GB)"],
    ),
    Case(
        id="unknown_cpu",
        prompt="Will a Ryzen 7 10800X3D bottleneck an RTX 5090?",
        include=[NOT_FOUND],
    ),
]
