"""Small statistics helpers used across the project's evaluations."""

from __future__ import annotations

import math


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default: 95%).

    Better behaved than the normal approximation for small samples and for rates near 0 or 1,
    which is where fall rates and task success rates usually sit.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == trials else min(1.0, centre + half)
    return lo, hi
