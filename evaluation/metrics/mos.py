from __future__ import annotations


class MOSModelUnavailable(RuntimeError):
    pass


def predict_mos(audio, sample_rate: int) -> float:
    """Use an installed third-party MOS predictor; never fabricate a MOS value."""
    try:
        import speechmos
    except ImportError as exc:
        raise MOSModelUnavailable(
            "Install a validated MOS predictor such as speechmos/UTMOS and pin its model "
            "version before reporting objective MOS."
        ) from exc
    if hasattr(speechmos, "predict"):
        return float(speechmos.predict(audio, sample_rate))
    raise MOSModelUnavailable("installed speechmos package has no supported predict() API")
