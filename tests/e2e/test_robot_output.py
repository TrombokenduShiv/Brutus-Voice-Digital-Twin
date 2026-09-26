import numpy as np

from voice_twin.device.robot_renderer import RobotRenderer


def test_robot_renderer_outputs_24k_and_safe_peak():
    renderer = RobotRenderer(target_sr=24000)
    audio = np.ones(48000, dtype=np.float32) * 2.0
    output = renderer.render(audio, 48000)
    assert 23990 <= len(output) <= 24010
    assert float(np.max(np.abs(output))) <= 0.951
