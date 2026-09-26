import numpy as np

from training.features import extract_utterance


def test_lightweight_feature_extraction_runs(tmp_path):
    import soundfile as sf

    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = 0.1 * np.sin(2 * np.pi * 180 * t)
    wav = tmp_path / "speaker.wav"
    sf.write(wav, audio, sr)

    item = extract_utterance(
        {
            "id": "u1",
            "speaker_id": "s1",
            "audio": str(wav),
            "text": "hello world",
            "language": "en-us",
            "session_id": "session-1",
        },
        aligner="proportional",
        lightweight=True,
    )
    assert len(item.token_ids) > 0
    assert item.prosody_target.shape[1] == 5
    assert item.accent_target.shape[1] == 8
    assert item.acoustic_target.shape[1] == 80
