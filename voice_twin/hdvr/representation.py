from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HDVRBatch:
    identity: torch.Tensor
    vocal: torch.Tensor
    speaker_memory: torch.Tensor
    accent: torch.Tensor
    prosody: torch.Tensor
    events: torch.Tensor

    def batch_size(self) -> int:
        return int(self.identity.shape[0])
