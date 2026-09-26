from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    metrics: dict[str, float] = field(default_factory=dict)
    passed: dict[str, bool] = field(default_factory=dict)
    missing_gates: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.passed) and all(self.passed.values()) and not self.missing_gates


def flatten_gates(tree: dict) -> dict[str, dict]:
    flat: dict[str, dict] = {}
    for key, value in tree.items():
        if isinstance(value, dict) and ("minimum" in value or "maximum" in value):
            flat[key] = value
        elif isinstance(value, dict):
            flat.update(flatten_gates(value))
    return flat


def apply_gates(
    metrics: dict[str, float],
    gates: dict,
    *,
    require_all: bool = False,
) -> EvaluationResult:
    flat = flatten_gates(gates)
    passed: dict[str, bool] = {}
    missing: list[str] = []
    for name, rule in flat.items():
        if name not in metrics:
            if require_all:
                missing.append(name)
            continue
        value = float(metrics[name])
        ok = True
        if "minimum" in rule:
            ok &= value >= float(rule["minimum"])
        if "maximum" in rule:
            ok &= value <= float(rule["maximum"])
        passed[name] = bool(ok)
    return EvaluationResult(metrics=metrics, passed=passed, missing_gates=missing)
