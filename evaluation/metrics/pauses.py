from __future__ import annotations

from voice_twin.prosody.pause_detector import Pause


def pause_metrics(reference: list[Pause], predicted: list[Pause], tolerance_ms: float = 100.0) -> dict[str, float]:
    used: set[int] = set()
    start_errors, duration_errors = [], []
    matches = 0
    for ref in reference:
        choices = [(i, abs(pred.start_s-ref.start_s)) for i, pred in enumerate(predicted) if i not in used]
        if not choices:
            continue
        i, err = min(choices, key=lambda item:item[1])
        if err * 1000 <= tolerance_ms:
            used.add(i)
            matches += 1
            start_errors.append(err*1000)
            duration_errors.append(abs(predicted[i].duration_s-ref.duration_s)*1000)
    precision = matches / len(predicted) if predicted else (1.0 if not reference else 0.0)
    recall = matches / len(reference) if reference else 1.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {
        "pause_precision": precision,
        "pause_recall": recall,
        "pause_f1": f1,
        "pause_start_mae_ms": sum(start_errors)/len(start_errors) if start_errors else 0.0,
        "pause_duration_mae_ms": sum(duration_errors)/len(duration_errors) if duration_errors else 0.0,
    }
