"""Vector scientific figures with explicit curve-level provenance."""
from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_figures_data import ordered_progress,sha

COLORS={'UF':'#555555','A':'#d17b14','B':'#258b54','C':'#9051a3','D':'#2467b0'}


class Figures:
    def __init__(self,data,metrics,out):
        self.data=data;self.metrics=metrics;self.out=Path(out);self.curves=[];self.events=[]
        (self.out/'figures').mkdir();(self.out/'plot_data').mkdir()
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                             'axes.titlesize':10,'legend.fontsize':8,'lines.linewidth':1.,
                             'path.simplify':False,'pdf.fonttype':42,'savefig.pad_inches':.06})
        for key,(d,p,e,path) in data.items():
            shutil.copy2(path/'executed.npz',self.out/'plot_data'/(key+'.npz'))

    def d(self,key):return self.data[key][0]

    def line(self,ax,figure,key,x,y,label,signals,transform,units,x_domain='physical_time',curve_kind=None,**kwargs):
        d,p,events,path=self.data[key]
        ax.plot(x,y,label=label,**kwargs)
        if curve_kind is None:
            if 'reserve' in signals:curve_kind='unexecuted_prediction'
            elif transform.startswith('E0+cumsum'):curve_kind='offline_diagnostic'
            elif 'diagnostic' in transform:curve_kind='counterfactual_diagnostic'
            elif any(s.startswith(('x_des','v_des','alpha_target')) for s in signals):curve_kind='online_reference'
            elif 'tick_seconds' in signals:curve_kind='measured_runtime'
            else:curve_kind='executed_observation_or_derived_work'
        self.curves.append(dict(figure=figure,curve=label,case=key,signals=signals,curve_kind=curve_kind,
            transformation=transform,units=units,
            evaluation_window_s=[float(x[0]),float(x[-1])] if x_domain=='physical_time' else [float(d['t_phys'][0]),float(d['t_phys'][-1]+d['h'])],
            x_axis=dict(domain=x_domain,units='s' if x_domain=='physical_time' else 'wall ms',range=[float(x[0]),float(x[-1])]),
            sampling_stage=p['stage'],source_commit=p['metadata']['commit'],
            numerical_source_sha256=p['metadata']['source_sha256'],
            source_kind=p['source_kind'],config=p['metadata']['case'],
            raw_file=str(path/'executed.npz'),raw_sha256=sha(path/'executed.npz'),
            portable_plot_data='plot_data/'+key+'.npz',
            raw_signal_map=p['raw_signal_map'],original_raw_files=p['source_files']))

    def axes(self,n,height,title):
        fig,axes=plt.subplots(n,1,figsize=(7.25,height),sharex=True,layout='constrained')
        axes=np.atleast_1d(axes)
        fig.suptitle(title,fontsize=12)
        for i,ax in enumerate(axes):
            ax.grid(alpha=.23);ax.set_axisbelow(True)
            ax.text(.008,.96,'('+chr(97+i)+')',transform=ax.transAxes,va='top',fontweight='bold',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=.3))
        axes[-1].set_xlabel('Physical simulation time (s)')
        return fig,axes

    def phase(self,axes,key,mark=True):
        case=self.data[key][1]['metadata']['case']
        profiles=case.get('profile') or [case]
        for j,p in enumerate(profiles):
            a=p['start'];r=p['ramp'];hold=p['hold'];end=a+2*r+hold
            helping=p.get('name')=='helping' or p['magnitude']<0
            color='#d7eadd' if helping else '#efd9e4'
            for ax in axes:
                ax.axvspan(a,end,color=color,alpha=.5,zorder=0)
                ax.axvline(a+r+hold,color='#8b5c73',ls=':',lw=.7)
                ax.axvline(end,color='#444444',ls='--',lw=.7)
            # Explicit event labels avoid making the two release anchors
            # distinguishable only by an unexplained line style.
            if j==len(profiles)-1:
                axes[-1].plot([],[],color='#8b5c73',ls=':',lw=.8,label='Ramp-out onset (dotted)')
                axes[-1].plot([],[],color='#444444',ls='--',lw=.8,label='Pulse end (dashed)')
                handles,labels=axes[-1].get_legend_handles_labels()
                axes[-1].legend(handles,labels,loc='best',fontsize=7,ncols=2)
            if mark:
                axes[0].text((a+end)/2,.89,'help pulse' if helping else 'block pulse',
                     transform=axes[0].get_xaxis_transform(),ha='center',fontsize=7,
                     bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=1))
            self.events.append(dict(case=key,onset_s=a,rampout_s=a+r+hold,pulse_end_s=end,
                                     source='frozen case.profile or case HumanPulse parameters'))

    def save(self,fig,name,axes,window=(8.71,18)):
        for ax in axes:ax.set_xlim(*window);ax.margins(y=.12)
        self.display_window(name,window)
        for ext in ('pdf','png'):fig.savefig(self.out/'figures'/(name+'.'+ext),dpi=180)
        plt.close(fig)

    def display_window(self,name,window):
        for curve in self.curves:
            if curve['figure']==name:
                curve['source_domain_s']=curve['evaluation_window_s']
                if curve['x_axis']['domain']=='physical_time':curve['displayed_window_s']=list(window)
                else:curve['displayed_x_range_wall_ms']=curve['x_axis']['range']

    def make(self):
        self.f3();self.f46();self.f5()
        if 'help-then-block__D' in self.data:self.f7()
        self.f8();self.supplement()
        (self.out/'curve_manifest.json').write_text(json.dumps(dict(curves=self.curves,events=self.events,
            convention='No smoothing, no decimation, no extrapolation. Energy at t+h; power at t. '
            'Reserve is an unexecuted shadow prediction, never an executed trajectory.',
            thresholds=dict(ledger_floor_J=0.,normal_force_target_N=10.,
                force_quality_band_N=[7.,13.],recovery_force_band_N=[9.5,10.5],
                recovery_alpha=.95,recovery_hold_s=.5,real_upper_energy_cap='not implemented',
                peak_human_power_cap='not implemented',source='frozen GuardConfig and source_semantics.md')),
                indent=2,allow_nan=False)+'\n')

    def f3(self):
        key='nohuman-zero__D';d=self.d(key);f='fig3_task_exemption'
        fig,a=self.axes(3,6.2,'F3 adapted: no-human task exemption; offline monitored-port diagnostic')
        E0=self.data[key][1]['metadata']['case']['budget'];te=d['t_phys']+d['h']
        self.line(a[0],f,key,te,d['energy_H'],'Online H = R (KUKA D; UF/A/B/C coincide)',
                  ['energy_H','energy_R'],'raw post-increment ledger','J',color=COLORS['D'])
        diagnostic=E0+np.cumsum(d['task_work']+d['work_H'])
        self.line(a[0],f,key,te,diagnostic,'Offline E0 + task + human work from t=0',
                  ['task_work','work_H'],'E0+cumsum(task_work+work_H); NOT a closed-loop policy','J',
                  color='#b05163',ls='--')
        a[0].axhline(0,color='k',lw=.6);a[0].set_ylabel('Energy (J)');a[0].legend(loc='lower left',fontsize=7)
        self.line(a[1],f,key,d['t_phys'],d['normal_force'],'Actual task-contact force',
                  ['normal_force'],'identity','N',color=COLORS['D'])
        a[1].axhline(10,color='k',ls='--',lw=.7,label='Original 10 N target');a[1].set_ylabel('Normal force (N)');a[1].legend(loc='upper right')
        shown=d['normal_force'][(d['t_phys']>=8.71-1e-9)&(d['t_phys']<18)]
        lo=min(10.,float(shown.min()));hi=max(10.,float(shown.max()));pad=max(.005,(hi-lo)*.15)
        a[1].set_ylim(lo-pad,hi+pad) # all extrema in the explicitly displayed task window
        self.line(a[2],f,key,d['t_control'],d['v'][:,0],'Actual TCP along-path velocity',
                  ['v[:,0]'],'world x component','m/s',color=COLORS['D'])
        self.line(a[2],f,key,d['t_control'],d['v_des'][:,0],'Online desired velocity',
                  ['v_des[:,0]'],'world x component','m/s',color='k',ls='--')
        a[2].set_ylabel('Tangential v (m/s)');a[2].legend(loc='lower left')
        a[0].text(.99,.96,'No actual intervention; floor = 0 J; no upper cap',ha='right',va='top',transform=a[0].transAxes,fontsize=7)
        for ax in a:ax.axvspan(8.71,18,color='#edf3ed',alpha=.3,zorder=0)
        self.save(fig,f,a)

    def f46(self):
        key='main__D';d=self.d(key);f='fig4_fig6_accounting'
        fig,a=self.axes(2,4.8,'F4 / F6 supported subset: identical human-work accounts')
        self.line(a[0],f,key,d['t_phys']+d['h'],d['energy_H'],'Online E_H',
                  ['energy_H'],'identity','J',color=COLORS['D'],lw=1.8)
        self.line(a[0],f,key,d['t_phys']+d['h'],d['energy_R'],'Online E_R = E_H',
                  ['energy_R'],'identity (coincident by implementation)','J',color='#b96491',ls='--')
        a[0].axhline(0,color='k',lw=.6);a[0].set_ylabel('Raw ledger (J)');a[0].legend(loc='upper right')
        self.line(a[1],f,key,d['t_phys'],d['power_H'],'Actual p_H = F_H dot v_H (into robot)',
                  ['power_H','force_H','velocity_H'],'identity; actual link5 material point, world coordinates','W',color=COLORS['D'])
        a[1].axhline(0,color='k',lw=.5);a[1].set_ylabel('Human power (W)');a[1].legend(loc='lower right')
        a[0].text(.02,.15,'Plant/predictor mismatch: not independently configured\np_delta and mismatch-aware residual: not implemented',
                  transform=a[0].transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        self.phase(a,key);self.save(fig,f,a)

    def f5(self):
        f='fig5_blocking';fig,a=self.axes(3,7.1,'F5 adapted: complete executed blocking work and raw energy')
        for v in ('UF','A','B','D'):
            key='main__'+v;d=self.d(key);r=self.metrics[key];color=COLORS[v]
            label='KUKA C = D' if v=='D' else 'KUKA '+v
            if not r['completed']:label+=' (STOP)'
            self.line(a[0],f,key,d['t_phys'],-d['power_H'],label,['power_H'],'negation; positive is robot-to-human output','W',color=color,alpha=.9)
            self.line(a[1],f,key,d['t_phys']+d['h'],-np.cumsum(d['work_H']),label,['work_H'],'-cumsum(signed physics increments)','J',color=color)
            self.line(a[2],f,key,d['t_phys']+d['h'],d['energy_H'],label,['energy_H'],'raw post-increment ledger, no lower clamp','J',color=color)
            if not r['completed']:
                for ax,y in zip(a,[-d['power_H'][-1],-np.sum(d['work_H']),d['energy_H'][-1]]):ax.plot(d['t_phys'][-1]+d['h'],y,'x',color=color,ms=6)
            if v in ('A','D') and r['first_intervention_s'] is not None:
                a[0].axvline(r['first_intervention_s'],color=color,ls='-.',lw=.8)
                a[0].text(r['first_intervention_s'],.98 if v=='A' else .72,v+' first change',rotation=90,fontsize=6,
                    transform=a[0].get_xaxis_transform(),ha='right',va='top')
        d=self.d('main__D')
        self.line(a[1],f,'main__D',d['t_phys']+d['h'],np.cumsum(np.maximum(-d['work_H'],0)),
            'KUKA D gross outgoing',['work_H'],'cumsum(max(-work_H,0)); distinct from net bound','J',color='#9d2476',ls='--')
        for ax in a:ax.axhline(0,color='k',lw=.5)
        a[0].set_ylabel('-p_H (W)');a[1].set_ylabel('Outgoing work (J)');a[2].set_ylabel('Raw E_H (J; symlog)')
        a[2].set_yscale('symlog',linthresh=1e-6)
        a[2].text(.99,.05,'Symlog exposes negative physics prefixes',ha='right',transform=a[2].transAxes,fontsize=7)
        a[0].legend(loc='upper right',ncols=2);a[1].legend(loc='lower right',fontsize=7,ncols=2)
        self.phase(a,'main__D');self.save(fig,f,a)

    def f7(self):
        key='help-then-block__D';d=self.d(key);f='fig7_charge_discharge'
        fig,a=self.axes(2,5.1,'F7 adapted: continuous helping then blocking (one state and ledger history)')
        self.line(a[0],f,key,d['t_phys']+d['h'],d['energy_H'],'Raw E_H = E_R; no finite cap',
                  ['energy_H','energy_R'],'raw uncapped ledger after every executed work increment','J',color=COLORS['D'])
        a[0].axhline(0,color='k',lw=.7,label='Implemented floor: 0 J');a[0].set_ylabel('Raw energy (J)');a[0].legend(loc='upper right')
        self.line(a[1],f,key,d['t_phys'],d['power_H'],'Measured p_H: positive charging, negative discharging',
                  ['power_H'],'identity; assess power sign, not force direction alone','W',color=COLORS['D'])
        a[1].axhline(0,color='k',lw=.5);a[1].set_ylabel('Human power (W)');a[1].legend(loc='upper right',fontsize=7)
        self.phase(a,key)
        r=self.metrics[key]
        if not r['completed']:
            for ax in a:ax.axvline(r['completed_s'],color='red',ls='-.',lw=1.)
            a[0].text(.99,.08,'STOP: '+str(r['stop_reason']),ha='right',transform=a[0].transAxes,fontsize=7)
        self.save(fig,f,a)

    def f8(self):
        f='fig8_recovery';fig,a=self.axes(6,8.4,'F8 adapted: interaction and release; coupled reserve/governor/backup')
        d=self.d('main__D');key='main__D'
        self.line(a[0],f,key,d['t_phys'],np.linalg.norm(d['force_H'],axis=1),'Measured link5 human force magnitude',
                  ['force_H'],'Euclidean norm of applied world force','N',color='#7d3b73')
        a[0].set_ylabel('|F_H| (N)');a[0].legend(loc='upper right',fontsize=8.5)
        first=self.metrics[key]['first_intervention_s']
        a[0].axvline(first,color=COLORS['D'],ls='-.',lw=.8)
        a[0].text(first,.05,'D first change',rotation=90,transform=a[0].get_xaxis_transform(),ha='right',fontsize=6)
        for v in ('UF','A','D'):
            k='main__'+v;x=self.d(k);color=COLORS[v]
            self.line(a[1],f,k,x['t_control'],1000*(x['x'][:,0]-x['x'][0,0]),v+' actual',
                      ['x[:,0]'],'1000*(x-x_initial); fixed spatial coordinate','mm',color=color)
            self.line(a[2],f,k,x['t_control'],x['v'][:,0],v+' actual',['v[:,0]'],'world x component','m/s',color=color)
            nominal=x.get('tau_ungoverned',x['tau_nom'])
            self.line(a[3],f,k,x['t_control'],np.max(np.abs(x['tau_exec']-nominal),axis=1),v+' total joint change',
                      ['tau_exec','tau_ungoverned' if 'tau_ungoverned' in x else 'tau_nom'],
                      'max_j abs(tau_exec-tau_ungoverned); A/UF original nominal is ungoverned','N m',color=color)
            self.line(a[4],f,k,x['t_phys'],x['normal_force']-10.,v+' normal-force error',
                      ['normal_force'],'F_n - ORIGINAL 10 N task target','N',color=color)
        self.line(a[1],f,key,d['t_control'],1000*(d['x_des'][:,0]-d['x'][0,0]),'D online governed reference',
                  ['x_des[:,0]'],'1000*(x_des-x_initial)','mm',color=COLORS['D'],ls='--')
        u=self.d('main__UF')
        self.line(a[1],f,'main__UF',u['t_control'],1000*(u['x_des'][:,0]-u['x'][0,0]),'UF time-indexed reference (diagnostic for D)',
                  ['x_des[:,0]'],'1000*(UF x_des - UF x_initial); paired run, NOT D online input','mm',color='k',ls=':',lw=1.1)
        self.line(a[2],f,key,d['t_control'],d['v_des'][:,0],'D online governed v_des',
                  ['v_des[:,0]'],'identity','m/s',color=COLORS['D'],ls='--')
        self.line(a[5],f,key,d['t_phys']+d['h'],d['energy_H'],'D E_H',
                  ['energy_H'],'raw post-increment ledger','J',color=COLORS['D'],lw=1.8)
        self.line(a[5],f,key,d['t_phys']+d['h'],d['energy_R'],'D E_R (identical)',
                  ['energy_R'],'raw post-increment ledger','J',color='#b96491',ls='--')
        a[5].axhline(0,color='k',lw=.6)
        for ax,label in zip(a[1:],['Along position (mm)','Tangential v (m/s)','Joint change (N m)','F_n - 10 N (N)','Raw ledger (J)']):ax.set_ylabel(label)
        a[4].axhspan(-3,3,color='#cbe5d1',alpha=.4);a[4].axhline(0,color='k',lw=.5)
        for ax in a:
            ax.yaxis.label.set_size(10)
            ax.tick_params(labelsize=9)
        for ax in a[1:]:ax.legend(loc='upper right',fontsize=8.5,ncols=2)
        a[3].text(.02,.08,'Panel (d) replaces the absent scalar Cartesian command',transform=a[3].transAxes,fontsize=7,
                    bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
        self.phase(a,key);self.save(fig,f,a)

    def supplement(self):
        key='main__D';d=self.d(key);f='figS_reserve_progress_timing'
        fig,a=self.axes(4,7.5,'KUKA supplement: reserve, progression, progress and full-tick timing')
        self.line(a[0],f,key,d['t_phys']+d['h'],d['energy_H'],'Executed E_H',
                  ['energy_H'],'identity','J',color=COLORS['D'])
        self.line(a[0],f,key,d['t_control'],d['reserve'],'Unexecuted held-force backup reserve',
                  ['reserve'],'identity; shadow forecast quantity, NOT executed work','J',color='#b25e7c',ls='--')
        a[0].set_ylabel('Energy / reserve (J)');a[0].legend(loc='upper right',fontsize=7)
        self.line(a[1],f,key,d['t_control'],d['alpha'],'Actual alpha',
                  ['alpha'],'identity','dimensionless',color=COLORS['D'])
        self.line(a[1],f,key,d['t_control'],d['alpha_target'],'Selected alpha target',
                  ['alpha_target'],'identity','dimensionless',color='#c06d31',ls='--')
        a[1].set_ylabel('Progress rate alpha');a[1].legend(loc='lower right',fontsize=7)
        for v in ('UF','A','D'):
            k='main__'+v;x=self.d(k);r,progress,count=ordered_progress(x['x'][:,0],x['strokes'],x['s_des'])
            self.line(a[2],f,k,x['t_control'],1000*progress,v+' measured ordered progress',
                      ['x[:,0]','strokes','s_des'],'ordered alternating 0.1 m frontier; completion at0.0995m; x_initial anchor','mm',color=COLORS[v])
        a[2].set_ylabel('Ordered progress (mm)');a[2].legend(loc='upper left',fontsize=7)
        # Use a separate axis for timing. This panel deliberately has a distinct
        # x variable (wall-time distribution), explicitly labelled and unshared.
        a[3].remove()
        a[3]=fig.add_subplot(4,1,4)
        for v in ('UF','A','D'):
            k='main__'+v;x=self.d(k);q=np.sort(x['tick_seconds'])*1000;y=np.arange(1,len(q)+1)/len(q)
            self.line(a[3],f,k,q,y,v+' entire executed tick ECDF',
                      ['tick_seconds'],'sort(tick_seconds)*1000 vs rank/N; no timing stages omitted','wall ms / fraction',x_domain='wall_tick_duration',color=COLORS[v])
        a[3].set_xscale('log');a[3].axvline(1,color='k',ls='--',lw=.8,label='1 ms deadline')
        a[3].set_xlabel('Whole successful control tick (wall ms; log scale)');a[3].set_ylabel('Cumulative fraction');a[3].grid(alpha=.2)
        a[3].legend(loc='lower right',fontsize=7,ncols=2)
        a[3].text(.008,.96,'(d)',transform=a[3].transAxes,va='top',fontweight='bold')
        self.phase(a[:3],key)
        for ax in a[:3]:ax.set_xlim(8.71,18);ax.margins(y=.1)
        self.display_window(f,(8.71,18))
        for ext in ('pdf','png'):fig.savefig(self.out/'figures'/(f+'.'+ext),dpi=180)
        plt.close(fig)
