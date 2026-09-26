import numpy as np

from voice_twin.prosody.pause_detector import detect_pauses


def test_pause_detection():
    sr=1000
    x=np.concatenate([np.ones(100)*0.1,np.zeros(200),np.ones(100)*0.1]).astype(np.float32)
    pauses=detect_pauses(x,sr,frame_ms=10,min_ms=50)
    assert pauses
    assert pauses[0].duration_s >= 0.18
