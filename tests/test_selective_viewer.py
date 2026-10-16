import numpy as np
import pytest
from scripts.selective_viewer import RecordedRun


def trajectory():
    t=np.array([0.,.25,.5]);q=np.arange(21).reshape(3,7).astype(float)
    return RecordedRun('case','K3',t,q,q,np.zeros((3,7)),np.zeros((3,3)),np.ones((3,3)),
                       np.zeros((3,3)),np.ones(3)*10,np.array([0.,.5]),np.ones((2,3)),np.ones(2),{},dict(completed_s=.75,completion='STOP'))


def test_seek_selects_actual_pose_without_interpolation():
    r=trajectory();frame=r.sample(.37)
    assert frame['time']==.25
    np.testing.assert_array_equal(frame['q'],r.q[1]);assert not frame['finished']


def test_failed_trial_freezes_at_last_recorded_pose():
    r=trajectory();frame=r.sample(100.)
    assert frame['time']==.5;assert frame['finished']
    np.testing.assert_array_equal(frame['q'],r.q[-1])
    assert r.sample(-100.)['time']==0.


def test_missing_full_logs_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError,match='No recorded joint poses'):
        RecordedRun.load(tmp_path,'case','K0')
