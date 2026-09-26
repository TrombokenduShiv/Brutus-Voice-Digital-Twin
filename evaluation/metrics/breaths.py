from __future__ import annotations


def event_f1(reference_starts: list[float], predicted_starts: list[float], tolerance_s: float = 0.15) -> dict[str,float]:
    remaining=set(range(len(predicted_starts))); matches=0
    for r in reference_starts:
        options=[(i,abs(predicted_starts[i]-r)) for i in remaining]
        if options:
            i,d=min(options,key=lambda x:x[1])
            if d<=tolerance_s:
                remaining.remove(i); matches+=1
    precision=matches/len(predicted_starts) if predicted_starts else (1.0 if not reference_starts else 0.0)
    recall=matches/len(reference_starts) if reference_starts else 1.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"breath_precision":precision,"breath_recall":recall,"breath_f1":f1}
