"""Interactive playback of saved KUKA comparisons, without rerunning physics.

macOS: .venv/bin/mjpython scripts/selective_viewer.py
Uses recorded joint poses, force and account samples. A stopped trial freezes
at its final recorded pose and is explicitly labelled STOP; no tail is invented.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import signal
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

MODES=['K0','K1','K2','K3','K4']
NAMES=['nominal','whole port','residual only','dual account / governor OFF','dual account / governor ON']


@dataclass
class RecordedRun:
    case: str
    mode: str
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    torque: np.ndarray
    force: np.ndarray
    energy: np.ndarray
    power: np.ndarray
    normal_force: np.ndarray
    ct: np.ndarray
    reference: np.ndarray
    alpha: np.ndarray
    meta: dict
    metrics: dict

    @classmethod
    def load(cls,root,case,mode):
        folder=Path(root)/case/mode
        if not (folder/'baseline_log.npz').exists():
            raise FileNotFoundError(f'No recorded joint poses for {case}/{mode}: {folder}/baseline_log.npz. Use --root with the full experiment logs.')
        meta=json.loads((folder/'metadata.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
        with np.load(folder/'baseline_log.npz',allow_pickle=False) as b:
            t,q,qd,torque=[b['phys__'+k] for k in ('t','q','qd','tau_applied')]
            ct,ref=b['ctl__t'],b['ctl__x_d']
        with np.load(folder/'selective.npz',allow_pickle=False) as z:
            force,energy,power,fn=[z['phys__'+k] for k in ('force','energy','power','normal_force')]
            alpha=z['ctl__alpha']
            if not np.array_equal(t,z['phys__t']):raise ValueError('pose/account timestamps differ')
        if len(t)==0 or np.any(np.diff(t)<=0):raise ValueError('empty or unordered recorded trajectory')
        return cls(case,mode,t,q,qd,torque,force,energy,power,fn,ct,ref,alpha,meta,metrics)

    def sample(self,cursor):
        """Use an actual left sample; never interpolate or extrapolate a trial."""
        i=int(np.clip(np.searchsorted(self.t,cursor,side='right')-1,0,len(self.t)-1))
        j=int(np.clip(np.searchsorted(self.ct,self.t[i],side='right')-1,0,len(self.ct)-1))
        return dict(index=i,control_index=j,time=float(self.t[i]),q=self.q[i],qd=self.qd[i],torque=self.torque[i],
                    force=self.force[i],energy=self.energy[i],power=self.power[i],normal_force=float(self.normal_force[i]),
                    alpha=float(self.alpha[j]),reference=self.reference[j],
                    finished=bool(cursor>=self.metrics['completed_s']-1e-10))


def draw_arrow(mujoco,scene,point,force):
    magnitude=float(np.linalg.norm(force))
    if magnitude<1e-6:return
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.95,.15,.6,1.]))
    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.01,point-force*.04,point)
    scene.ngeom+=1


def set_pose(mujoco,model,data,run,frame):
    data.qpos[:]=frame['q'];data.qvel[:]=frame['qd'];data.ctrl[:]=frame['torque'];data.time=frame['time']
    data.qfrc_applied[:]=0
    # Forward evaluation reconstructs rendering geometry, never advances time.
    mujoco.mj_forward(model,data)
    body=model.body(run.meta['case']['body']).id
    return data.xpos[body]+data.xmat[body].reshape(3,3)@np.array(run.meta['case']['local'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT/'results/selective/final')
    ap.add_argument('--case',default='C_blocking');ap.add_argument('--mode',choices=MODES,default='K4')
    ap.add_argument('--start',type=float,default=9.2);ap.add_argument('--speed',type=float,default=1.)
    ap.add_argument('--play',action='store_true',help='Start playing instead of paused')
    ap.add_argument('--max-wall-seconds',type=float,help='Close automatically after this many seconds (GUI smoke check)')
    ap.add_argument('--check',action='store_true',help='Validate the selected recorded trajectory without a GUI')
    args=ap.parse_args()
    if not np.isfinite(args.start) or not np.isfinite(args.speed) or args.speed<=0:ap.error('finite start and positive speed required')
    cases=sorted(p.name for p in args.root.iterdir() if p.is_dir() and any((p/m/'baseline_log.npz').exists() for m in MODES))
    if args.case not in cases:ap.error(f'case must be one of {cases}')
    cache={}
    def load(case,mode):
        key=(case,mode)
        if key not in cache:
            cache.clear()  # Keep only the displayed trajectory in memory.
            cache[key]=RecordedRun.load(args.root,case,mode)
        return cache[key]
    run=load(args.case,args.mode)
    if args.check:
        f=run.sample(args.start);print(json.dumps(dict(case=run.case,mode=run.mode,samples=len(run.t),first_s=float(run.t[0]),last_s=float(run.t[-1]),selected_s=f['time'],outcome=run.metrics['completion'])));return 0
    import mujoco
    import mujoco.viewer
    from kuka_table_sim.scene import load_model
    model=load_model();data=mujoco.MjData(model)
    cursor=max(0.,args.start);speed=args.speed;paused=not args.play;stop=False;keys=queue.SimpleQueue();notice=''
    try:old_handler=signal.signal(signal.SIGINT,lambda *_:keys.put(ord('Q')))
    except ValueError:old_handler=None
    print('Recorded comparison viewer: 0-4 select K0-K4 | P pause/play | R restart at 9.2 s | arrows seek 0.25 s | N/B next/previous case | comma/period slower/faster | Q quit. Mouse orbits/pans/zooms.',flush=True)
    started=time.perf_counter();previous=started
    try:
        with mujoco.viewer.launch_passive(model,data,key_callback=keys.put,show_left_ui=False,show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:]=[.4,.08,.3];viewer.cam.distance=1.8;viewer.cam.azimuth=140;viewer.cam.elevation=-20
                groups=viewer.opt.geomgroup.copy()
            while viewer.is_running() and not stop:
                now=time.perf_counter();elapsed=min(now-previous,.1);previous=now
                if args.max_wall_seconds and now-started>=args.max_wall_seconds:break
                if not paused:cursor+=elapsed*speed
                while not keys.empty():
                    key=keys.get()
                    if ord('0')<=key<=ord('4'):
                        mode='K'+chr(key)
                        try:run=load(run.case,mode);notice=''
                        except FileNotFoundError:notice=f'{mode} was not tested in {run.case}'
                        with viewer.lock():viewer.opt.geomgroup[:]=groups
                    elif key in (ord('P'),ord('p'),32):paused=not paused
                    elif key in (ord('R'),ord('r')):cursor=max(0.,args.start);paused=True
                    elif key==262:cursor+=.25
                    elif key==263:cursor=max(0.,cursor-.25)
                    elif key in (ord('N'),ord('n'),ord('B'),ord('b')):
                        direction=1 if key in (ord('N'),ord('n')) else -1
                        case=cases[(cases.index(run.case)+direction)%len(cases)]
                        mode=run.mode if (args.root/case/run.mode/'baseline_log.npz').exists() else 'K4'
                        run=load(case,mode);cursor=max(0.,args.start);paused=True;notice=''
                    elif key==ord(','):speed=max(.125,speed/2)
                    elif key==ord('.'):speed=min(8.,speed*2)
                    elif key in (ord('Q'),ord('q')):stop=True
                frame=run.sample(cursor)
                if frame['finished']:paused=True
                status='STOPPED - final recorded pose' if frame['finished'] and run.metrics['completion']=='STOP' else ('END - final recorded pose' if frame['finished'] else ('PAUSED' if paused else 'PLAYING'))
                with viewer.lock():
                    point=set_pose(mujoco,model,data,run,frame)
                    viewer.user_scn.ngeom=0;draw_arrow(mujoco,viewer.user_scn,point,frame['force'])
                    reference=frame['reference'][0]
                    if np.isfinite(reference):
                        tcp=data.site_xpos[model.site('tcp').id];g=viewer.user_scn.geoms[viewer.user_scn.ngeom]
                        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.009,0,0]),np.array([reference,tcp[1],tcp[2]+.025]),np.eye(3).ravel(),np.array([1.,.72,.1,1.]));viewer.user_scn.ngeom+=1
                E=frame['energy'];F=frame['force']
                left='Recorded playback\nController\nStatus\nTimeline / sample\nTrial end / outcome\nNormal force\nHuman force [N]\nH / R / W [J]\nHuman outgoing power\nGovernor alpha'
                right=f"{run.case}\n{run.mode}: {NAMES[int(run.mode[1])]}\n{status} ({speed:g}x)\n{cursor:.3f} / {frame['time']:.3f} s\n{run.metrics['completed_s']:.3f} s / {run.metrics['completion']}\n{frame['normal_force']:.3f} N\n{F[0]:+.2f}, {F[1]:+.2f}, {F[2]:+.2f}\n{E[0]:+.5f} / {E[1]:+.5f} / {E[2]:+.5f}\n{-frame['power'][0]:+.5f} W\n{frame['alpha']:.3f}"
                hints='0-4 controller | P pause/play | R restart | arrows seek\nN/B case | ,/. speed | Q quit\nMagenta: human force | Gold: tangential reference\nRecorded poses; no resimulation.'
                viewer.set_texts([(None,mujoco.mjtGridPos.mjGRID_TOPLEFT,left,right),(None,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,hints,notice)])
                viewer.sync();time.sleep(1/60)
    finally:
        if old_handler is not None:signal.signal(signal.SIGINT,old_handler)
    print('Recorded viewer closed.',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
