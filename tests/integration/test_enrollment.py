import numpy as np
import soundfile as sf

from voice_twin.enrollment.enrollment import EnrollmentPipeline


def test_enrollment_prepare(tmp_path):
    p=tmp_path/"a.wav"
    sf.write(p,np.sin(np.linspace(0,100,24000)).astype(np.float32)*0.1,24000)
    sample=EnrollmentPipeline().prepare(p)
    assert sample.sample_rate==24000
    assert sample.quality.duration_s > 0.9
