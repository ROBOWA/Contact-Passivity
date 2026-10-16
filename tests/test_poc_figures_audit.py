"""Numerical honesty checks for the independent report audit."""
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts'/'poc_figures_audit.py'
spec=importlib.util.spec_from_file_location('independent_poc_audit',PATH)
audit=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=audit
spec.loader.exec_module(audit)


def test_signed_work_distinguishes_recharge_net_and_gross():
    # World-frame force acts on the human application point: initial positive
    # F.v charges; subsequent negative F.v delivers energy to the person.
    force=np.array([[2.,0,0],[2.,0,0]])
    velocity=np.array([[.5,0,0],[-1.,0,0]])
    metrics,energy=audit.signed_work(np.einsum('ij,ij->i',force,velocity),.1,.12)
    assert metrics['human_work_in_J']==pytest.approx(-.1)
    assert metrics['net_outgoing_J']==pytest.approx(.1)
    assert metrics['gross_outgoing_J']==pytest.approx(.2)
    assert metrics['gross_incoming_J']==pytest.approx(.1)
    assert energy==pytest.approx([.22,.02])


def test_short_negative_prefix_remains_visible_after_endpoint_recovers():
    metrics,energy=audit.signed_work([-2.,2.,0.,0.],.001,.001)
    assert energy[-1]==pytest.approx(.001)
    assert metrics['min_substep_energy_J']==pytest.approx(-.001)
    assert metrics['gross_outgoing_J']==pytest.approx(.002)


def test_recovery_does_not_pad_early_termination():
    t=np.arange(10)*.001
    assert audit.sustained_recovery(t,np.ones(10,bool),.008,.001,.005) is None
    assert audit.sustained_recovery(t,np.ones(10,bool),.02,.001,.005) is None
    assert audit.sustained_recovery(t,np.ones(10,bool),.005,.001,.005)==pytest.approx(0.)


def test_recovery_requires_sustained_observations():
    good=np.array([1,1,0,1,1,1,1,1,0,1],bool)
    assert audit.sustained_recovery(np.arange(10)*.001,good,0.,.001,.005)==pytest.approx(.003)


def test_dropped_sample_rejected_in_full_trace():
    with pytest.raises(audit.IntegrityError,match='dropped'):
        audit.validate_time([0.,.001,.003],.001)
    # A compact trace's index gap is explicit; it does not change quadrature dt.
    assert audit.validate_time([0.,.001,.003],.001,[0,1,3])==pytest.approx(0.)


def test_nonfinite_power_is_rejected():
    with pytest.raises(audit.IntegrityError,match='nonfinite'):
        audit.signed_work([0.,np.nan],.001,.12)


def _compact_case(path,fail=False):
    path.mkdir()
    n=8;dt=.00025;t=np.arange(n)*dt;power=np.zeros(n)
    metadata=dict(case=dict(name='early',budget=.12,start=.001,ramp=.001,hold=.001,
                           body='link5',local=[0,.04,.12]),variant='D',duration=.02,
                  config=dict(task_start=0.))
    (path/'metadata.json').write_text(json.dumps(metadata))
    np.savez(path/'trace.npz',t=t,power=power,velocity=np.zeros((n,3)),force=np.zeros((n,3)),
             normal_force=np.full(n,10.),present=np.ones(n,bool),energy=np.full((n,2),.12),
             physics_indices=np.arange(n),total_physics_count=n,
             ctl__t=np.array([0.,.001]),ctl__alpha=np.ones(2))
    with gzip.open(path/'events.json.gz','wt') as f:json.dump([],f)
    if fail:(path/'failure.json').write_text(json.dumps(dict(t=.002,reason='explicit_stop')))
    return path


def test_early_stop_is_reported_at_last_executed_step(tmp_path):
    result=audit.audit_case(_compact_case(tmp_path/'raw',True))
    m=result['metrics']
    assert result['status']=='PASS' # A correct audit can describe a stopped run.
    assert m['completion']=='STOPPED'
    assert m['completed_s']==pytest.approx(.002)
    assert m['interaction_complete'] is False
    assert m['force_recovery_s'] is None
    assert m['stop_reason']=='explicit_stop'


def test_solver_records_are_not_reserve_candidate_counts(tmp_path):
    folder=tmp_path/'events';folder.mkdir()
    events=[dict(t=1.,attempts=[dict(retry=0,status='solved',accepted=True,affine_margin=[0.,0.])]),
            dict(t=2.,attempts=[dict(retry=0,status='failed',accepted=False,reason='energy_infeasible')]),
            dict(t=3.,rejections=[dict(target=0.,reason='reserve_budget')])]
    with gzip.open(folder/'events.json.gz','wt') as f:json.dump(events,f)
    counts=audit.audit_events(folder)
    assert counts['qp_attempt_records']==2
    assert counts['qp_solves']==1
    assert counts['qp_active_row_count']==2
    assert counts['reserve_candidate_rejections']=={'reserve_budget':1}


def test_cli_reports_integrity_errors_with_nonzero_real_exit_status(tmp_path):
    case=_compact_case(tmp_path/'bad')
    with np.load(case/'trace.npz') as z:arrays={k:z[k] for k in z.files}
    arrays['t'][3]+=.000125
    np.savez(case/'trace.npz',**arrays)
    output=tmp_path/'audit.json'
    result=subprocess.run([sys.executable,str(PATH),'--case',str(case),'--out',str(output)],
                          capture_output=True,text=True,timeout=20)
    assert result.returncode==2
    report=json.loads(output.read_text())
    assert report['status']=='FAIL'
    assert report['counts']['errors']==1


def test_common_window_comparison_does_not_invent_a_completion_value():
    values=audit.compare_reported({'force_rmse_N':2.},{'force_rmse_N':2.,'completed':False},
                                 {'force_rmse_N':'force_rmse_N'})
    assert values['force_rmse_N']['status']=='PASS'
    assert 'completed' not in values


def test_normalized_bundle_retains_a_stopped_negative_prefix(tmp_path):
    bundle=tmp_path/'bundle';folder=bundle/'early__D';folder.mkdir(parents=True)
    dt=.00025;t=np.arange(8)*dt;tc=np.array([0.,.001])
    power=np.array([-8.,8.,0,0,0,0,0,0]);energy=.001+np.cumsum(power*dt)
    arrays=dict(t_phys=t,h=np.array(dt),t_control=tc,power_H=power,work_H=power*dt,
        energy_H=energy,energy_R=energy.copy(),force_H=np.column_stack((power,np.zeros((8,2)))),
        velocity_H=np.column_stack((np.ones(8),np.zeros((8,2)))),point_H=np.zeros((8,3)),
        normal_force=np.full(8,10.),present=np.ones(8,bool),task_power=np.zeros(8),task_work=np.zeros(8),
        tcp_velocity=np.zeros((8,3)),tcp_position=np.zeros((8,3)),x=np.zeros((2,3)),v=np.zeros((2,3)),
        x_des=np.zeros((2,3)),v_des=np.zeros((2,3)),force_des=np.full(2,10.),phase=np.zeros(2),
        strokes=np.zeros(2),s_des=np.zeros(2),tau_exec=np.zeros((2,7)),tau_nom=np.zeros((2,7)),
        tau_previous=np.zeros((2,7)),alpha=np.ones(2))
    np.savez(folder/'executed.npz',**arrays)
    provenance=dict(metadata=dict(case=dict(name='early',budget=.001,start=.001,ramp=.001,hold=.001),
                    variant='D',duration=.02,config=dict(task_start=0.)),
                    archived_metrics=dict(failure=dict(t=.002,reason='explicit_stop')),
                    original_directory=str(tmp_path/'source_not_available'),
                    output_sha256=audit.sha256(folder/'executed.npz'))
    (folder/'provenance.json').write_text(json.dumps(provenance))
    with gzip.open(folder/'events.json.gz','wt') as f:json.dump([],f)
    (bundle/'catalog.json').write_text(json.dumps(dict(cases=[dict(key='early__D',folder='early__D')])))
    reported=tmp_path/'reported.json'
    reported.write_text(json.dumps({'early__D':dict(min_substep_energy_J=-.001,completed=False,
                        force_recovery_from_pulse_end_s=None,stop_reason='explicit_stop')}))
    result=audit.audit_bundle(bundle,reported)
    assert result['status']=='PASS'
    assert result['cases'][0]['metrics']['min_substep_energy_J']==pytest.approx(-.001)
    assert result['cases'][0]['metrics']['completed_s']==pytest.approx(.002)
    assert result['cases'][0]['metrics']['force_recovery_s'] is None
    # A portable audit still checks physical identities when the original raw
    # directory is unavailable; a valid hash alone is insufficient evidence.
    arrays['energy_R']=arrays['energy_R']+.01
    np.savez(folder/'executed.npz',**arrays)
    provenance['output_sha256']=audit.sha256(folder/'executed.npz')
    (folder/'provenance.json').write_text(json.dumps(provenance))
    wrong=audit.audit_bundle(bundle,reported)
    assert wrong['status']=='FAIL'
    assert wrong['cases'][0]['checks']['account_difference_J']==pytest.approx(.01)
