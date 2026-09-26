from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evaluation.report import write_report
from evaluation.runner import apply_gates
from evaluation.suites.accent import evaluate_accent
from evaluation.suites.cloning import evaluate_clone
from evaluation.suites.performance import evaluate_performance
from evaluation.suites.robot import evaluate_robot
from evaluation.suites.unseen_text import evaluate_unseen_text


SUITES = {
    "cloning": evaluate_clone,
    "performance": evaluate_performance,
    "accent": evaluate_accent,
    "robot": evaluate_robot,
    "unseen": evaluate_unseen_text,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BRUTUS Voice Digital Twin audio.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="cloning")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--self-reference")
    parser.add_argument("--reference-text")
    parser.add_argument("--hypothesis-text")
    parser.add_argument("--heavy-identity", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gates",
        default="configs/evaluation/success_gates.yaml",
    )
    parser.add_argument("--require-all-gates", action="store_true")
    parser.add_argument("--output", default="artifacts/reports/evaluation.json")
    args = parser.parse_args()

    if args.suite in {"cloning", "performance", "unseen"}:
        metrics = SUITES[args.suite](
            args.reference,
            args.generated,
            reference_text=args.reference_text,
            hypothesis_text=args.hypothesis_text,
            self_reference_path=args.self_reference,
            heavy_identity=args.heavy_identity,
            device=args.device,
        )
    elif args.suite == "robot":
        metrics = evaluate_robot(args.reference, args.generated)
    else:
        metrics = evaluate_accent(args.reference, args.generated)

    gates = yaml.safe_load(Path(args.gates).read_text(encoding="utf-8"))["success_gates"]
    result = apply_gates(metrics, gates, require_all=args.require_all_gates)
    payload = {
        "suite": args.suite,
        "metrics": result.metrics,
        "passed": result.passed,
        "missing_gates": result.missing_gates,
        "success": result.success,
    }
    write_report(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
