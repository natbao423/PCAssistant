"""Pricing and value tools.

Everything that reads ``current_prices.csv`` and computes price, MSRP
markup, and performance-per-dollar lives here:

* ``check_current_value`` — one component's price and value.
* ``compare_components`` — two components side by side (single tool call,
  so small models don't have to remember to look up each one separately).
* ``find_best_value`` — ranks a category under a budget.

Out-of-stock entries are represented in the CSV as ``-1`` or a marker
string like ``"OOS"``; ``_parse_price`` normalises both to a sentinel.
"""

import pandas as pd

from ..config import PRICE_PATH
from ..data import find_price_row, find_row, load_cpu_df, load_gpu_df
from ..matching import ambiguity_note

_OOS_SENTINEL = -1.0
_OOS_STRINGS = {"oos", "out of stock", "n/a", "none", "unavailable", "-", ""}


def _parse_price(raw: object) -> float:
    """Safely convert a price cell to a float.

    Handles numeric values, numeric strings, and common out-of-stock
    markers such as ``"OOS"``, ``"N/A"``, ``"None"``, ``"-"``, and
    ``"unavailable"``.

    Args:
        raw: The value read from the ``Current_Price_USD`` column.

    Returns:
        The price as a float, or ``-1.0`` if the item is out of stock
        or the value cannot be parsed.
    """
    if pd.isna(raw):
        return _OOS_SENTINEL
    if isinstance(raw, (int, float)):
        return float(raw)
    cleaned = str(raw).strip().lstrip("$").replace(",", "")
    if cleaned.lower() in _OOS_STRINGS:
        return _OOS_SENTINEL
    try:
        return float(cleaned)
    except ValueError:
        return _OOS_SENTINEL


def _normalise_retailer(raw: object) -> str:
    """Return a display-friendly retailer name from a CSV cell.

    Treats ``"None"``, ``NaN``, and empty strings as unknown.  All other
    values (``"Amazon"``, ``"Newegg"``, ``"Best Buy"``, ``"Other"``, etc.)
    are returned as-is.

    Args:
        raw: The value read from the ``Retailer`` column.

    Returns:
        Retailer display string, or ``"Market"`` as a fallback.
    """
    if pd.isna(raw):
        return "Market"
    text = str(raw).strip()
    if text.lower() in ("none", ""):
        return "Market"
    return text


def _get_price_and_specs(
    component_name: str,
) -> tuple[str | None, pd.Series | None, float | None, str | None, bool, str]:
    """Resolve the current price and spec row for a CPU or GPU.

    Searches both the GPU and CPU DataFrames for the model named by
    *component_name*, preferring an exact name match if the GPU and CPU
    tables both have a candidate.  If a ``current_prices.csv`` file exists
    the street price is preferred; otherwise the MSRP from the spec sheet
    is used.

    Args:
        component_name: Model name to look up, e.g. ``"RTX 5070 Ti"``.

    Returns:
        Tuple of ``(component_type, spec_row, price, price_source,
        is_out_of_stock, note)``, where ``note`` flags an ambiguous name
        (or is empty).  Values are ``None``/``False``/``""`` when the
        component is not found.
    """
    gpu_hit = find_row(load_gpu_df(), component_name)
    cpu_hit = find_row(load_cpu_df(), component_name)

    if gpu_hit and (cpu_hit is None or gpu_hit[1].exact or not cpu_hit[1].exact):
        (spec, mention), comp_type = gpu_hit, "GPU"
    elif cpu_hit:
        (spec, mention), comp_type = cpu_hit, "CPU"
    else:
        return None, None, None, None, False, ""
    note = ambiguity_note(component_name, mention)

    msrp = float(spec["MSRP_USD"])
    current_price: float | None = None
    source = "MSRP"
    is_oos = False

    if PRICE_PATH.exists():
        row = find_price_row(pd.read_csv(PRICE_PATH), spec["Model"])
        if row is not None:
            parsed = _parse_price(row["Current_Price_USD"])
            if parsed == _OOS_SENTINEL:
                is_oos = True
                current_price = msrp
                source = "MSRP (currently out of stock)"
            else:
                current_price = parsed
                source = _normalise_retailer(row.get("Retailer"))
        else:
            current_price = msrp
            source = "MSRP (no current price data)"
    else:
        current_price = msrp
        source = "MSRP (no price database)"

    return comp_type, spec, current_price, source, is_oos, note


def check_current_value(component_name: str) -> str:
    """Look up the current street price and value metrics for a component.

    Calculates the MSRP markup/discount and a performance-per-dollar
    score.  Falls back to MSRP when no live price is available.

    Args:
        component_name: GPU or CPU model name.

    Returns:
        Human-readable price and value summary, or a not-found message.
    """
    comp_type, spec, current_price, source, is_oos, note = _get_price_and_specs(
        component_name,
    )
    if spec is None:
        return (
            f"'{component_name}' NOT FOUND in any database.  "
            f"Do NOT guess — tell the user this component is not in the database."
        )

    msrp = float(spec["MSRP_USD"])
    markup = ((current_price - msrp) / msrp) * 100 if msrp > 0 else 0

    result = note + f"{spec['Model']}\n"
    if is_oos:
        result += f"⚠️ Currently OUT OF STOCK — using MSRP ${msrp:.0f} for calculations\n"
    else:
        result += f"Current price: ${current_price:.0f} ({source})\n"
        if source not in ("MSRP (no current price data)", "MSRP (no price database)"):
            direction = "above" if markup > 0 else "below"
            result += f"MSRP: ${msrp:.0f} ({markup:+.0f}% {direction} MSRP)\n"
        else:
            result += "Using MSRP — no current street price available\n"

    perf_col = "Relative_Perf_4K" if comp_type == "GPU" else "Gaming_Perf_1080p"
    perf_label = "4K gaming" if comp_type == "GPU" else "1080p gaming"
    perf = spec.get(perf_col, None)

    if perf is not None and str(perf).strip() != "-":
        perf = float(perf)
        value = perf / current_price * 100
        result += f"{perf_label} score: {perf}/100\n"
        result += f"Value: {value:.2f} perf/$ (higher is better)\n"

    return result


def compare_components(component_a: str, component_b: str) -> str:
    """Compare two GPUs or CPUs side-by-side on price, performance, and value.

    Fetches specs and pricing for both components in a single call and
    returns a formatted comparison.  Use this whenever the user asks to
    compare, contrast, or choose between two specific components.

    Args:
        component_a: First component name (e.g. ``"RTX 5080"``).
        component_b: Second component name (e.g. ``"RX 9070 XT"``).

    Returns:
        Side-by-side comparison string with specs, pricing, and a
        value verdict, or an error if either component is not found.
    """
    type_a, spec_a, price_a, src_a, oos_a, note_a = _get_price_and_specs(component_a)
    type_b, spec_b, price_b, src_b, oos_b, note_b = _get_price_and_specs(component_b)

    if spec_a is None:
        return (
            f"'{component_a}' NOT FOUND in database.  "
            f"Do NOT guess specs — tell the user this component is not available."
        )
    if spec_b is None:
        return (
            f"'{component_b}' NOT FOUND in database.  "
            f"Do NOT guess specs — tell the user this component is not available."
        )

    # ── Build per-component summaries ───────────────────────────────
    def _summarise(
        spec: pd.Series,
        comp_type: str,
        price: float,
        source: str,
        is_oos: bool,
    ) -> dict:
        msrp = float(spec["MSRP_USD"])
        markup = ((price - msrp) / msrp) * 100 if msrp > 0 else 0

        if comp_type == "GPU":
            perf_col, perf_label = "Relative_Perf_4K", "4K gaming"
        else:
            perf_col, perf_label = "Gaming_Perf_1080p", "1080p gaming"

        perf_raw = spec.get(perf_col, None)
        perf = (
            float(perf_raw)
            if perf_raw is not None and str(perf_raw).strip() != "-"
            else None
        )
        value = (perf / price * 100) if perf and price and price > 0 else None

        return {
            "model": spec["Model"],
            "type": comp_type,
            "msrp": msrp,
            "price": price,
            "source": source,
            "is_oos": is_oos,
            "markup": markup,
            "perf": perf,
            "perf_label": perf_label,
            "value": value,
        }

    a = _summarise(spec_a, type_a, price_a, src_a, oos_a)
    b = _summarise(spec_b, type_b, price_b, src_b, oos_b)

    # ── Format output ───────────────────────────────────────────────
    def _fmt_one(s: dict) -> str:
        lines = [s["model"]]
        if s["is_oos"]:
            lines.append(f"  Price: OUT OF STOCK (MSRP ${s['msrp']:.0f})")
        else:
            lines.append(f"  Price: ${s['price']:.0f} ({s['source']})")
            lines.append(f"  MSRP: ${s['msrp']:.0f} ({s['markup']:+.0f}%)")
        if s["perf"] is not None:
            lines.append(f"  {s['perf_label']} score: {s['perf']}/100")
        if s["value"] is not None:
            lines.append(f"  Value: {s['value']:.2f} perf/$")
        return "\n".join(lines)

    output = note_a + note_b + f"=== {a['model']} vs {b['model']} ===\n\n"
    output += _fmt_one(a) + "\n\n"
    output += _fmt_one(b) + "\n\n"

    # ── Verdict ─────────────────────────────────────────────────────
    if a["value"] is not None and b["value"] is not None:
        if a["value"] > b["value"]:
            diff = ((a["value"] - b["value"]) / b["value"]) * 100
            output += (
                f"Value winner: {a['model']} — {diff:.0f}% better perf per dollar.\n"
            )
        elif b["value"] > a["value"]:
            diff = ((b["value"] - a["value"]) / a["value"]) * 100
            output += (
                f"Value winner: {b['model']} — {diff:.0f}% better perf per dollar.\n"
            )
        else:
            output += "Value: tied on perf per dollar.\n"

    if a["perf"] is not None and b["perf"] is not None:
        if a["perf"] > b["perf"]:
            diff = ((a["perf"] - b["perf"]) / b["perf"]) * 100
            output += (
                f"Performance winner: {a['model']} — "
                f"{diff:.0f}% faster ({a['perf']} vs {b['perf']}).\n"
            )
        elif b["perf"] > a["perf"]:
            diff = ((b["perf"] - a["perf"]) / a["perf"]) * 100
            output += (
                f"Performance winner: {b['model']} — "
                f"{diff:.0f}% faster ({b['perf']} vs {a['perf']}).\n"
            )
        else:
            output += "Performance: tied.\n"

    return output


def find_best_value(budget: float, component_type: str) -> str:
    """Rank components under a budget by performance-per-dollar.

    Iterates over every model of the given type, pairs benchmark scores
    with current (or MSRP) prices, and returns the top seven by value.

    Args:
        budget:         Maximum price in USD.
        component_type: ``"GPU"`` or ``"CPU"``.

    Returns:
        A numbered list of recommendations, or a not-found message.
    """
    if component_type.upper() == "GPU":
        spec_df = load_gpu_df()
        perf_col = "Relative_Perf_4K"
    else:
        spec_df = load_cpu_df()
        perf_col = "Gaming_Perf_1080p"

    price_df = (
        pd.read_csv(PRICE_PATH) if PRICE_PATH.exists()
        else pd.DataFrame()
    )

    results: list[dict] = []
    for _, spec in spec_df.iterrows():
        perf = spec.get(perf_col, None)
        if perf is None or str(perf).strip() == "-":
            continue
        perf = float(perf)

        model = spec["Model"]
        price: float | None = None
        source = "MSRP"
        is_oos = False

        price_row = find_price_row(price_df, model)
        if price_row is not None:
            parsed = _parse_price(price_row["Current_Price_USD"])
            if parsed == _OOS_SENTINEL:
                is_oos = True
            else:
                price = parsed
                source = "current"

        if price is None and not is_oos:
            price = float(spec["MSRP_USD"])
            source = "MSRP"

        if is_oos or price is None or price <= 0 or price > budget:
            continue

        results.append({
            "model": model,
            "price": price,
            "source": source,
            "perf": perf,
            "value": perf / price * 100,
        })

    if not results:
        return f"No {component_type}s with benchmark data found under ${budget:.0f}."

    results.sort(key=lambda x: x["value"], reverse=True)

    lines = [f"Best value {component_type}s under ${budget:.0f} (by perf per dollar):\n"]
    for i, r in enumerate(results[:7], 1):
        price_note = f"${r['price']:.0f}"
        if r["source"] == "MSRP":
            price_note += " (MSRP, no current price)"
        lines.append(
            f"{i}. {r['model']} — {price_note} | "
            f"perf: {r['perf']}/100 | value: {r['value']:.2f}"
        )

    return "\n".join(lines)
