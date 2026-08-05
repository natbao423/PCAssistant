"""CPU-GPU bottleneck analysis tool.

Compares a CPU's 1080p gaming score against the minimum recommended score
for the GPU's performance tier, then returns verdicts for both 1080p and
4K. Tier thresholds live in ``config.BOTTLENECK_TIERS``.
"""

from ..config import BOTTLENECK_TIERS
from ..data import load_cpu_df, load_gpu_df


def _classify_gpu_tier(gpu_score: float) -> tuple[float, str]:
    """Map a GPU's 4K performance score to its tier and minimum CPU score.

    Args:
        gpu_score: Relative 4K performance score (0–100, RTX 5090 = 100).

    Returns:
        Tuple of ``(minimum_cpu_score, tier_label)``.
    """
    for floor, min_cpu, label in BOTTLENECK_TIERS:
        if gpu_score >= floor:
            return min_cpu, label
    # Fallback (should not be reached given the 0-floor entry above).
    return 60.0, "budget"


def _format_bottleneck_1080p(
    cpu_model: str,
    cpu_score: float,
    min_cpu: float,
) -> str:
    """Generate the 1080p bottleneck verdict string.

    Args:
        cpu_model: Display name of the CPU.
        cpu_score: CPU gaming benchmark score (0–100).
        min_cpu:   Minimum recommended CPU score for the GPU tier.

    Returns:
        A single emoji-prefixed verdict paragraph.
    """
    headroom = cpu_score - min_cpu

    if headroom < -10:
        return (
            f"🔴 Significant CPU bottleneck at 1080p. The {cpu_model} "
            f"(score {cpu_score}) is well below the {min_cpu} minimum for "
            f"this GPU tier. Expect 15-30% lost performance in CPU-heavy "
            f"games. Upgrade strongly recommended."
        )
    if headroom < 0:
        return (
            f"🟡 Mild CPU bottleneck at 1080p. The {cpu_model} "
            f"(score {cpu_score}) is slightly below the {min_cpu} minimum "
            f"for this GPU tier. You may lose 5-15% in CPU-intensive "
            f"titles. Playable, but not ideal."
        )
    if headroom < 8:
        return (
            f"🟢 Adequate pairing. The {cpu_model} (score {cpu_score}) "
            f"meets the minimum for this GPU tier. No major bottleneck, "
            f"but little headroom."
        )
    return (
        f"✅ Well-balanced pairing. The {cpu_model} (score {cpu_score}) "
        f"comfortably exceeds the {min_cpu} minimum for this GPU tier."
    )


def _format_bottleneck_4k(gpu_model: str, gpu_score: float) -> str:
    """Generate the 4K resolution assessment paragraph.

    Args:
        gpu_model: Display name of the GPU.
        gpu_score: Relative 4K performance score (0–100).

    Returns:
        A short paragraph describing 4K behaviour.
    """
    if gpu_score >= 50:
        return (
            "At 4K: GPU-limited in most games — CPU bottleneck is minimal. "
            "This pairing works better at 4K than 1080p."
        )
    if gpu_score >= 25:
        return "At 4K: Mostly GPU-limited. CPU bottleneck is minor at this resolution."
    return f"At 4K: The {gpu_model} will struggle at 4K regardless of CPU."


def check_cpu_gpu_bottleneck(cpu_name: str, gpu_name: str) -> str:
    """Analyse whether a CPU–GPU pairing has a bottleneck at 1080p and 4K.

    Looks up both components by name, compares the CPU gaming score against
    the minimum threshold for the GPU's performance tier, and returns a
    human-readable verdict.

    Args:
        cpu_name: CPU model name (or substring), e.g. ``"Ryzen 5 5600X"``.
        gpu_name: GPU model name (or substring), e.g. ``"RTX 5080"``.

    Returns:
        Multi-line analysis string with scores, tier info, and verdicts
        for both 1080p and 4K, or an error message if either component
        is not found.
    """
    cpu_df = load_cpu_df()
    gpu_df = load_gpu_df()

    cpu_match = cpu_df[cpu_df["Model"].str.contains(cpu_name, case=False, na=False)]
    gpu_match = gpu_df[gpu_df["Model"].str.contains(gpu_name, case=False, na=False)]

    if cpu_match.empty:
        return (
            f"CPU '{cpu_name}' NOT FOUND in database.  "
            f"Do NOT guess — tell the user this CPU is not in the database."
        )
    if gpu_match.empty:
        return (
            f"GPU '{gpu_name}' NOT FOUND in database.  "
            f"Do NOT guess — tell the user this GPU is not in the database."
        )

    cpu = cpu_match.iloc[0]
    gpu = gpu_match.iloc[0]
    cpu_score = float(cpu["Gaming_Perf_1080p"])
    gpu_perf = gpu.get("Relative_Perf_4K", None)

    if gpu_perf is None or str(gpu_perf).strip() == "-":
        return (
            f"CPU: {cpu['Model']} (gaming score: {cpu_score}/100)\n"
            f"GPU: {gpu['Model']} (no 4K benchmark available)\n"
            f"Cannot determine bottleneck without GPU benchmark data."
        )

    gpu_score = float(gpu_perf)
    min_cpu, tier = _classify_gpu_tier(gpu_score)

    header = (
        f"CPU: {cpu['Model']} — gaming score: {cpu_score}/100 (9800X3D = 100)\n"
        f"GPU: {gpu['Model']} — 4K score: {gpu_score}/100 (RTX 5090 = 100)\n"
        f"GPU tier: {tier} — minimum recommended CPU score: {min_cpu}\n"
    )
    verdict_1080p = _format_bottleneck_1080p(cpu["Model"], cpu_score, min_cpu)
    verdict_4k = _format_bottleneck_4k(gpu["Model"], gpu_score)

    return f"{header}\n{verdict_1080p}\n\n{verdict_4k}"
