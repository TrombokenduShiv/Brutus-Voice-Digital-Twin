import numpy as np

from voice_twin.audio.normalize import peak_normalize,to_pcm16


def test_peak_and_pcm():
    x=peak_normalize(np.array([0.0,2.0,-2.0],dtype=np.float32))
    assert np.max(np.abs(x)) <= 0.951
    assert len(to_pcm16(x)) == len(x)*2
