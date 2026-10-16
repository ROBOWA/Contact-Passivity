import numpy as np
import pytest
from scripts.selective_video import frame_times


def test_normal_speed_sixty_second_video_has_exact_frame_count():
    t=frame_times(0,60,30,60.0000000002)
    assert len(t)==1800
    assert t[0]==0
    assert t[-1]==pytest.approx(60-1/30)
    np.testing.assert_allclose(np.diff(t),1/30,atol=1e-14)


def test_video_cannot_invent_a_stopped_trial_tail():
    with pytest.raises(ValueError,match='beyond the recorded trial'):
        frame_times(0,60,30,10.754)


def test_video_window_keeps_actual_simulation_timestamps():
    t=frame_times(9.2,60,30,60.)
    assert t[0]==9.2
    assert len(t)==1524
    assert t[-1]<60
