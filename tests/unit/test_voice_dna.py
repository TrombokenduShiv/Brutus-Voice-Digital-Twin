import numpy as np

from voice_twin.profiles.voice_dna import VoiceDNA


def test_voice_dna_roundtrip():
    p=VoiceDNA("s",np.ones(4),np.ones(3),np.ones((2,4)))
    q=VoiceDNA.from_jsonable(p.to_jsonable())
    assert q.speaker_id=="s"
    assert q.identity_memory.shape==(2,4)
