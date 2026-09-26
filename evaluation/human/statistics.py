from __future__ import annotations

from math import sqrt


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float,float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    p=successes/trials
    denom=1+z*z/trials
    center=(p+z*z/(2*trials))/denom
    half=z*sqrt((p*(1-p)+z*z/(4*trials))/trials)/denom
    return center-half, center+half


def real_fake_equivalent(successes: int, trials: int, margin: float = 0.05) -> bool:
    lo,hi=wilson_interval(successes,trials)
    return lo >= 0.5-margin and hi <= 0.5+margin
