from __future__ import annotations

import math


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, fn)
