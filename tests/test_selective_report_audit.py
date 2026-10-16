"""The exported audit must reject corrupted executed evidence."""
import copy
import numpy as np
from scripts.selective_report import audit


def fixture_run():
    h=.00025;power=np.tile([-.1,-.2,-.4],(4,1));initial=np.array([.12,.3,.3])
    energy=initial+np.cumsum(power*h,axis=0)
    p=dict(power=power,energy=energy,discarded=np.zeros((4,3)),force=np.tile([-1.,0,0],(4,1)),
        human_velocity=np.tile([.1,0,0],(4,1)),human_tau=np.tile([-1.,0,0,0,0,0,0],(4,1)),
        qd=np.tile([.1,0,0,0,0,0,0],(4,1)),task_force=np.tile([-3.,0,0],(4,1)),
        task_point_velocity=np.tile([.1,0,0],(4,1)),task_moment=np.zeros((4,3)),task_omega=np.zeros((4,3)),
        task_hat_force=np.tile([-2.,0,0],(4,1)),task_hat_point_velocity=np.tile([.1,0,0],(4,1)),
        task_hat_moment=np.zeros((4,3)),task_hat_omega=np.zeros((4,3)),
        energy_before=np.vstack([initial,energy[:-1]]),torque=np.zeros((4,7)),prediction_error=np.zeros((4,2)),unexpected=np.zeros(4))
    c=dict(t=np.zeros(1),tau_exec=np.zeros((1,7)),tau_nom=np.zeros((1,7)),tau_previous=np.zeros((1,7)))
    return dict(p=p,c=c,m=dict(qp_attempts=0,net_outgoing_J=[.0001,.0002,.0004],gross_outgoing_J=[.0001,.0002,.0004]),
        meta=dict(mode='K3',source_sha256='same',source_end_sha256='same',config=dict(initial=initial.tolist(),caps=[.15,.3,.3],floors=[0,0,0],safe_energy_J=2.5e-5,barrier_rate=5.,peak_human_W=.15)),
        bm=dict(torque_limit=[100]*7),events=[])


def test_audit_accepts_physical_accounting():
    assert audit(fixture_run())['status']=='PASS'


def test_audit_rejects_corrupted_cap_ledger_and_power_sign():
    run=fixture_run();run['p']['energy'][1,0]+=.01
    assert audit(run)['status']=='FAIL'
    run=fixture_run();run['p']['human_tau']*=-1
    assert audit(run)['status']=='FAIL'


def test_audit_retains_failed_qp_attempt_count():
    run=fixture_run();run['events']=[dict(failed=True,attempts=[dict(accepted=False,reason='solver_failure')])]
    assert audit(run)['status']=='FAIL'
    run['m']['qp_attempts']=1
    assert audit(run)['status']=='PASS'
