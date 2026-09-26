from __future__ import annotations

from voice_twin.conversion.performance_transfer import PerformanceTransfer, PerformanceTransferRequest


def replicate_performance(transfer: PerformanceTransfer, request: PerformanceTransferRequest):
    return transfer.synthesize(request)
