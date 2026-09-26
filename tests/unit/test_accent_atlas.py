from voice_twin.accent.atlas import AccentAtlas


def test_running_mean():
    a=AccentAtlas()
    a.update("t","final",100)
    a.update("t","final",200)
    assert a.get("t","final").duration_ms_mean == 150
    assert a.to_dict()["t|final"]["count"] == 2
