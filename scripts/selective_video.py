"""Render a normal-speed MP4 directly from a saved comparison trajectory.

No integration/controller calls: recorded poses and measurements are reused.
An external ffmpeg or the optional imageio-ffmpeg package provides H.264.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.selective_viewer import RecordedRun,draw_arrow,set_pose,NAMES


def frame_times(start,end,fps,completed):
    if not all(np.isfinite(v) for v in (start,end,fps,completed)) or start<0 or end<=start or fps<=0:
        raise ValueError('finite increasing nonnegative time range and positive fps required')
    if end>completed+1e-8:raise ValueError('requested video would extend beyond the recorded trial')
    return start+np.arange(int(math.ceil((end-start)*fps-1e-9)))/fps


class Layout:
    def __init__(self,width,height,run,start,end):
        import matplotlib
        fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
        self.width,self.height,self.run,self.start,self.end=width,height,run,start,end
        self.scale=width/1920.;self.scene_w=int(width*.72)//2*2;self.scene_h=int(height*.77)//2*2
        self.scene_y=int(100*self.scale)
        self.fonts={key:ImageFont.truetype(str(fonts/name),max(10,int(size*self.scale))) for key,size,name in (
            ('title',40,'DejaVuSans-Bold.ttf'),('small',22,'DejaVuSans.ttf'),('label',22,'DejaVuSans.ttf'),
            ('value',31,'DejaVuSans.ttf'),('time',52,'DejaVuSans-Bold.ttf'),('badge',24,'DejaVuSans-Bold.ttf'))}

    def compose(self,rgb,frame):
        s=self.scale;w,h=self.width,self.height;run=self.run
        canvas=Image.new('RGB',(w,h),'#f5f6f8');canvas.paste(Image.fromarray(rgb),(0,self.scene_y));d=ImageDraw.Draw(canvas)
        def text(x,y,value,font='small',fill='#263344'):
            d.text((int(x*s),int(y*s)),value,font=self.fonts[font],fill=fill)
        text(38,18,f'KUKA iiwa  |  {run.mode}: {NAMES[int(run.mode[1])]}','title')
        text(40,72,'Recorded simulation  |  Normal speed (1x)  |  TCP blocking')
        x=1415;y=125;t=frame['time'];force=float(np.linalg.norm(frame['force']))
        if force>.02 or 9.5<=t<=10.9:phase='HUMAN FORCE ACTIVE';color='#c51c76'
        elif t>10.9:phase='CONTINUED SLIDING';color='#087f6a'
        else:phase='BEFORE INTERACTION';color='#526278'
        d.rounded_rectangle((int((x-10)*s),int(y*s),int(1890*s),int((y+55)*s)),radius=int(10*s),fill=color)
        text(x,138,phase,'badge','white')
        text(x,210,f'{t:05.2f} s','time');text(x,274,f'of {self.end:.0f} s recorded simulation')
        rows=[('Normal force / target',f"{frame['normal_force']:.2f} N / 10 N"),
              ('Human force',f'{force:.2f} N'+('  (-x)' if frame['force'][0]<-.02 else '')),
              ('Human outgoing power',f"{-frame['power'][0]:+.4f} W  /  0.15 W limit"),
              ('Human energy H',f"{frame['energy'][0]:+.5f} J"),
              ('Residual energy R',f"{frame['energy'][1]:+.5f} J"),
              ('Whole-port W (monitored)',f"{frame['energy'][2]:+.5f} J"),
              ('Reference speed alpha',f"{frame['alpha']:.3f}")]
        for n,(label,value) in enumerate(rows):
            yy=325+n*82
            text(x,yy,label,'label','#607086');text(x,yy+29,value,'value')
        text(40,947,'Magenta arrow: measured human force  |  Gold dot: tangential reference')
        text(40,980,'Same 5 N interaction at 9.5-10.9 s, followed by extended back-and-forth sliding.')
        left,right=45,1875;bar_y=1027
        d.line((int(left*s),int(bar_y*s),int(right*s),int(bar_y*s)),fill='#c3cad4',width=max(2,int(5*s)))
        def xx(v):return int((left+(right-left)*np.clip((v-self.start)/(self.end-self.start),0,1))*s)
        d.line((xx(9.5),int(bar_y*s),xx(10.9),int(bar_y*s)),fill='#c51c76',width=max(3,int(11*s)))
        d.line((xx(self.start),int(bar_y*s),xx(t),int(bar_y*s)),fill='#087f6a',width=max(2,int(4*s)))
        d.ellipse((xx(t)-int(7*s),int((bar_y-7)*s),xx(t)+int(7*s),int((bar_y+7)*s)),fill='#087f6a')
        text(43,1040,f'{self.start:g} s');text(1760,1040,f'{self.end:g} s')
        return np.asarray(canvas)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT/'results/selective/long60');ap.add_argument('--case',default='C_blocking')
    ap.add_argument('--mode',default='K4',choices=['K0','K1','K2','K3','K4']);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--start',type=float,default=0.);ap.add_argument('--end',type=float)
    ap.add_argument('--fps',type=int,default=30);ap.add_argument('--width',type=int,default=1920);ap.add_argument('--height',type=int,default=1080)
    ap.add_argument('--ffmpeg',type=Path);ap.add_argument('--preview-only',action='store_true')
    a=ap.parse_args()
    if a.width<960 or a.height<540 or a.width%2 or a.height%2 or abs(a.width/a.height-16/9)>1e-6:ap.error('even 16:9 dimensions at least 960x540 required')
    if a.out.exists():raise FileExistsError(a.out)
    run=RecordedRun.load(a.root,a.case,a.mode);end=round(run.metrics['completed_s'],8) if a.end is None else a.end
    times=frame_times(a.start,end,a.fps,run.metrics['completed_s']);layout=Layout(a.width,a.height,run,a.start,end)
    if a.case!='C_blocking' or a.mode!='K4':ap.error('this video layout currently describes the C_blocking K4 experiment')
    if a.ffmpeg:ffmpeg=str(a.ffmpeg)
    else:
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError as exc:raise RuntimeError('Provide --ffmpeg PATH or install imageio-ffmpeg in your video environment') from exc
    import mujoco
    from kuka_table_sim.scene import load_model
    model=load_model();model.vis.global_.offwidth=max(model.vis.global_.offwidth,layout.scene_w);model.vis.global_.offheight=max(model.vis.global_.offheight,layout.scene_h)
    data=mujoco.MjData(model);camera=mujoco.MjvCamera()
    camera.lookat[:]=[.4,.08,.36];camera.distance=1.95;camera.azimuth=140;camera.elevation=-20
    options=mujoco.MjvOption()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    renderer=mujoco.Renderer(model,height=layout.scene_h,width=layout.scene_w)
    def render(t):
        frame=run.sample(t);point=set_pose(mujoco,model,data,run,frame)
        renderer.update_scene(data,camera=camera,scene_option=options)
        # Lift the illustration clear of the tabletop; a thin leader preserves
        # the actual force application point instead of hiding the arrow in it.
        raised=point+np.array([0.,0.,.06])
        draw_arrow(mujoco,renderer.scene,raised,frame['force'])
        if np.linalg.norm(frame['force'])>1e-6:
            geom=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.95,.15,.6,1.]))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,2.,point,raised);renderer.scene.ngeom+=1
        if np.isfinite(frame['reference'][0]):
            tcp=data.site_xpos[model.site('tcp').id];geom=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.009,0,0]),np.array([frame['reference'][0],tcp[1],tcp[2]+.025]),np.eye(3).ravel(),np.array([1.,.72,.1,1.]));renderer.scene.ngeom+=1
        return layout.compose(renderer.render(),frame),frame
    if a.preview_only:
        rgb,_=render(a.start);Image.fromarray(rgb).save(a.out);renderer.close();return 0
    partial=a.out.with_name(a.out.stem+'.partial.mp4')
    if partial.exists():raise FileExistsError(partial)
    command=[ffmpeg,'-n','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{a.width}x{a.height}','-r',str(a.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)]
    started=time.perf_counter();encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE);indices=[]
    try:
        with a.out.with_suffix('.frames.csv').open('x') as fp:
            writer=csv.writer(fp,lineterminator='\n');writer.writerow(['frame','requested_s','recorded_index','recorded_s','normal_force_N','human_force_x_N','human_power_W','H_J','R_J','W_J','alpha'])
            for n,t in enumerate(times):
                rgb,frame=render(float(t));encoder.stdin.write(rgb.tobytes());indices.append(frame['index'])
                writer.writerow([n,float(t),frame['index'],frame['time'],frame['normal_force'],float(frame['force'][0]),float(frame['power'][0]),*frame['energy'].tolist(),frame['alpha']])
                if n==int(round((10.4-a.start)*a.fps)):Image.fromarray(rgb).save(a.out.with_suffix('.poster.png'))
                if n==0 or (n+1)%(a.fps*5)==0 or n==len(times)-1:print(f'Rendered {n+1}/{len(times)} frames ({t-a.start:.1f}/{end-a.start:.1f} s)',flush=True)
    finally:
        renderer.close();encoder.stdin.close()
    error=encoder.stderr.read().decode('utf-8',errors='replace');code=encoder.wait()
    if code:raise RuntimeError(f'ffmpeg exit {code}: {error}')
    partial.rename(a.out)
    verification=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-i',str(a.out),'-map','0:v:0','-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True)
    progress=dict(line.split('=',1) for line in verification.stdout.splitlines() if '=' in line)
    sources={name:hashlib.sha256((a.root/a.case/a.mode/name).read_bytes()).hexdigest() for name in ('baseline_log.npz','selective.npz','metadata.json','metrics.json')}
    manifest=dict(case=a.case,mode=a.mode,source_root=str(a.root.resolve()),source_hashes=sources,
        source_commit=run.meta['commit'],numerical_source_sha256=run.meta['source_sha256'],
        start_s=a.start,end_s=end,normal_speed=True,fps=a.fps,frames=len(times),size=[a.width,a.height],
        encoding='H.264 / yuv420p / CRF20 / faststart / no audio',ffmpeg_version=subprocess.check_output([ffmpeg,'-version'],text=True).splitlines()[0],
        video_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),frame_pose_sequence_sha256=hashlib.sha256(run.q[indices].tobytes()).hexdigest(),
        decode_exit_code=verification.returncode,decoded_frames=int(progress.get('frame',0)),
        decode_seconds=float(progress.get('out_time_us',0))/1e6,render_wall_seconds=time.perf_counter()-started,
        sampling='Actual recorded left samples; no pose interpolation, no controller or integration calls',
        force_arrow_display_offset_m=[0.,0.,.06],force_arrow_leader='Thin line links raised illustration to the actual force application point')
    a.out.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if verification.returncode or manifest['decoded_frames']!=len(times):raise RuntimeError('Video decode/frame-count verification failed')
    print(json.dumps(manifest),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
