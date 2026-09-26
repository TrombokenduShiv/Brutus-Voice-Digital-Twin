from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    metrics: dict[str,float] = field(default_factory=dict)
    passed: dict[str,bool] = field(default_factory=dict)


def apply_gates(metrics: dict[str,float], gates: dict[str,dict]) -> dict[str,bool]:
    out={}
    for name,rule in gates.items():
        if name not in metrics:
            continue
        value=metrics[name]
        ok=True
        if "minimum" in rule: ok &= value >= rule["minimum"]
        if "maximum" in rule: ok &= value <= rule["maximum"]
        out[name]=bool(ok)
    return out
