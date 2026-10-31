"""One fresh IPC episode: archived targets plus fixed/frozen gain control.

No training, object pose writes, live VLM, hidden support controller or checkpoint
restore. Each process starts at home and replays the full physical prefix.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from .core import Trajectory, interval_overlap
from .soft_scene import SoftScene
from .soft_taxel import SoftTaxelAdapter
from .soft_contact import IPCContactAudit


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def state_frame(q, qd, mid, half, trajectory, tick, start_tick, end_tick, gain):
    """Exactly the existing frozen actor's 79-field state contract."""
    now=tick/300
    nominal=trajectory.target(now)
    phase=trajectory.phase(now)
    return np.r_[(q-mid)/half, np.clip(qd/3,-10,10), (nominal-q)/.2,
                 (trajectory.target(now+1/15)-nominal)/.1, gain-1, gain-1,
                 [phase==x for x in ('preload','test_lift','lift','hold')],
                 (tick-start_tick)/(end_tick-start_tick)].astype(np.float32)


def gain_update(previous, action):
    requested=.6*np.exp((np.clip(float(action),-1,1)+1)*.5*np.log(5))
    return float(np.clip(previous+np.clip(requested-previous,-.2,.2),.6,3))


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',{'status':'initializing'})
    trajectory=Trajectory(args.trajectory)
    metadata=json.loads(args.trajectory.with_name('trajectory_metadata.json').read_text())
    assert trajectory.sha256==metadata['trajectory_sha256']
    assert metadata['scene']['scenario_name']=='large'
    assert np.allclose(metadata['scene']['target_initial_position_m'],[-.45,-.0333,.225])
    start_tick=int(math.floor(trajectory.phase_starts['preload']*15))/15*300
    start_tick=int(round(start_tick))
    end_tick=int(math.ceil(trajectory.duration_s*15))*20
    planned_end=end_tick
    if args.stop_time is not None:end_tick=min(end_tick,int(round(args.stop_time*300)))
    model=None;model_info=None
    torch.set_num_threads(4)
    if args.model:
        model_info={'path':str(args.model.resolve()),'sha256':hashlib.sha256(args.model.read_bytes()).hexdigest()}
        model=PPO.load(args.model,device='cpu')
        assert model.num_timesteps==786432, 'Unexpected training checkpoint'
        assert model.observation_space['state'].shape==(12,79)
        assert model.observation_space['tactile'].shape==(12,5,120,3)
        assert model.action_space.shape==(1,)
        assert model.policy.features_extractor.use_tactile==('no_tactile' not in args.controller)
    else:
        if args.fixed_gain is None:raise ValueError('Specify model or fixed gain')
    manifest={'controller':args.controller,'model':model_info,'mass_kg':args.mass_kg,
        'friction':args.friction,'training':False,'trajectory':str(args.trajectory.resolve()),
        'trajectory_sha256':trajectory.sha256,'start_tick':start_tick,'end_tick':end_tick,
        'planned_end_tick':planned_end,'hold_interval_s':[trajectory.phase_starts['hold'],trajectory.duration_s],
        'sensor_semantics':'Original virtual spring/damper formula on actual deforming FEM boundary, gated by physical finger/target contact existence; estimated nominal N.',
        'independent_contact':'IPC incremental-potential gradient / substep_dt^2; normal support separate from virtual taxels.',
        'load_metric':'Independent IPC normal primitive load only. Friction-gradient export is not validated and is not used as a complete force measurement.',
        'strict_definition':'Task plus >=2 sensor-bearing links with independent normal IPC load>.05N in all required hold samples.',
        'rates_hz':{'physics':300,'tactile':60,'action':15},'substeps':args.substeps,
        'newton_tolerance':args.newton_tolerance,
        'validity_sampling_hz':{'FEM_element_J':300,'rigid_collision':60},
        'recovery':'Fresh complete scene for every run; no IPC checkpoint or object state writes.',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__),*map(lambda n:Path(__file__).with_name(n),
                            ['soft_scene.py','soft_geometry.py','soft_coupling.py','soft_taxel.py','soft_contact.py','core.py','adaptive_policy.py'])]}}
    write(out/'manifest.json',manifest)
    samples=[];actions=[];obs_states=[];obs_touch=[];gain=1.;scene=None
    touch_history=np.zeros((12,5,120,3),np.float32);state_history=None
    fault=False;min_J_all=1.;last_print=time.monotonic();started=last_print
    prefix_saved=False;previous_phase=None;interval_min_J=1.
    try:
        scene=SoftScene(out/'environment',args.mass_kg,args.friction,
                        substeps=args.substeps,maxvolume=args.maxvolume,render=args.render,
                        newton_tolerance=args.newton_tolerance)
        assert scene.names==metadata['joint_order']
        adapter=SoftTaxelAdapter(scene.layouts)
        contact=IPCContactAudit(scene.scene,scene.soft,robot=scene.robot,table=scene.table,
            finger_links=dict(zip([l['finger'] for l in scene.layouts],scene.finger_link_objects)))
        write(out/'contact_mapping.json',{'records':contact.records,'friction':contact.friction_readback()})
        x=scene.rest_vertices[scene.tets]
        D=np.stack([x[:,i]-x[:,3] for i in range(3)],axis=-1)
        inv_D=np.linalg.inv(D);vol=np.abs(np.linalg.det(D))/6
        surface_ids=np.unique(scene.surface_triangles)
        if args.render:scene.capture('000000_initial')
        write(out/'status.json',{'status':'running','tick':0,'sim_time_s':0.})
        for tick in range(end_tick):
            if tick%20==0:
                if tick>=start_tick:
                    assert state_history is not None
                    if not prefix_saved:
                        np.savez_compressed(out/'prefix_observation.npz',state=state_history,
                            tactile=touch_history,vertices=scene.vertices(),q=scene.joints()[0])
                        prefix_saved=True
                    obs={'state':state_history[None].copy(),'tactile':touch_history[None].copy()}
                    if model is not None:
                        with torch.device('cpu'):
                            value,_=model.predict(obs,deterministic=True)
                        action=float(value.reshape(-1)[0])
                    else:action=float(2*np.log(args.fixed_gain/.6)/np.log(5)-1)
                    gain=gain_update(gain,action);scene.set_gain(gain)
                    actions.append([tick/300,action,gain])
                    obs_states.append(state_history.copy());obs_touch.append(touch_history.copy())
                scene.apply_target(trajectory.target(tick/300))
            scene.scene.step()
            vertices=scene.vertices().astype(float)
            if not np.isfinite(vertices).all():raise FloatingPointError('Nonfinite FEM nodes')
            x=vertices[scene.tets]
            J=np.linalg.det(np.stack([x[:,i]-x[:,3] for i in range(3)],axis=-1)@inv_D)
            if not np.isfinite(J).all():raise FloatingPointError('Nonfinite FEM element J')
            current_min=float(J.min());min_J_all=min(min_J_all,current_min)
            interval_min_J=min(interval_min_J,current_min)
            if current_min<=0:raise FloatingPointError('Inverted FEM element')
            if (tick+1)%5==0:
                now=(tick+1)/300
                poses=scene.finger_poses()
                ipc_transforms=np.stack([scene.scene.sim.coupler._abd_data_by_link[link][0].transform.copy()
                                         for link in scene.finger_link_objects])
                truth=contact.read()
                contact_mask=np.array([truth['fingers'][layout['finger']]['normal_primitive_count']>0
                                       for layout in scene.layouts])
                sensor=adapter.sample(vertices,scene.surface_triangles,poses['positions'],poses['rotations'],
                                      contact_mask=contact_mask)
                if not sensor['finite']:raise FloatingPointError('Nonfinite virtual tactile signal')
                force=np.asarray(sensor['force_link_N'],np.float32)
                q,qd=scene.joints()
                state=state_frame(q,qd,scene.mid,scene.half,trajectory,tick+1,start_tick,planned_end,gain)
                if not np.isfinite(state).all():raise FloatingPointError('Nonfinite actor state')
                touch_history[:-1]=touch_history[1:];touch_history[-1]=force
                if state_history is None:state_history=np.repeat(state[None],12,axis=0)
                else:state_history[:-1]=state_history[1:];state_history[-1]=state
                fault=fault or scene.rigid_collision_fault()
                sample={'time_s':now,'q':q,'qd':qd,'state':state,'gain':gain,
                    'tactile':force,'taxel_penetration_m':sensor['penetration_m'],
                    'taxel_active_count':sensor['active'].sum(axis=-1),
                    'finger_positions':poses['positions'],'finger_rotations':poses['rotations'],
                    'finger_ipc_transforms':ipc_transforms,
                    'sensor_contact_mask':contact_mask,
                    'contact_normal_load_N':np.asarray(truth['finger_normal_load_N']),
                    'clearance_m':float(vertices[surface_ids,2].min()-.2),
                    'min_J':interval_min_J,'volume_ratio':float((J*vol).sum()/vol.sum()),
                    'vertices':vertices.astype(np.float32),'fault':fault}
                samples.append(sample);interval_min_J=1.
                phase=trajectory.phase(now)
                if phase!=previous_phase and args.render:scene.capture(f'{tick+1:06d}_{phase}')
                previous_phase=phase
                if time.monotonic()-last_print>20:
                    progress={'controller':args.controller,'tick':tick+1,'sim_time_s':now,
                        'phase':phase,'clearance_m':sample['clearance_m'],'gain':gain,
                        'max_taxel_N':float(np.linalg.norm(force,axis=-1).max()),
                        'normal_load_N':sample['contact_normal_load_N'].tolist(),
                        'min_J':min_J_all,'wall_seconds':time.monotonic()-started}
                    print(json.dumps(progress),flush=True);write(out/'status.json',{'status':'running',**progress})
                    last_print=time.monotonic()
        if args.render:scene.capture(f'{end_tick:06d}_final')
        arrays={k:np.asarray([r[k] for r in samples]) for k in samples[0]}
        arrays.update(actions=np.asarray(actions),state_before=np.asarray(obs_states),tactile_before=np.asarray(obs_touch))
        np.savez_compressed(out/'trace.npz',**arrays)
        t=arrays['time_s'];hold_start=trajectory.phase_starts['hold'];hold_end=trajectory.duration_s
        weights=interval_overlap(t-1/60,t,hold_start,hold_end);mask=weights>1e-12
        support=(arrays['contact_normal_load_N']>.05).sum(-1)>=2
        hold_duration=float(weights.sum());full=abs(hold_duration-3)<1e-7
        task=bool(full and (arrays['clearance_m'][mask]>=.05).all() and not fault and min_J_all>0)
        contact_positive=arrays['contact_normal_load_N']>.05
        taxel_positive=np.linalg.norm(arrays['tactile'],axis=-1).max(-1)>1e-5
        misses=(contact_positive&~taxel_positive).sum(axis=0)
        counts=contact_positive.sum(axis=0)
        policy=t>start_tick/300
        score={'controller':args.controller,'status':'completed','mass_kg':args.mass_kg,'friction':args.friction,
            'task_success':task,'strict_success':bool(task and support[mask].all()),
            'hold_duration_s':hold_duration,'hold_min_clearance_m':float(arrays['clearance_m'][mask].min()) if mask.any() else None,
            'hold_supported_fraction':float((weights*support).sum()/hold_duration) if hold_duration else None,
            'final_clearance_m':float(arrays['clearance_m'][-1]),'final_supported':bool(support[-1]),
            'max_clearance_m':float(arrays['clearance_m'].max()),'min_J_all_physics_steps':min_J_all,
            'collision':fault,'sample_count':len(samples),'actions':len(actions),
            'mean_normal_load_N':float(arrays['contact_normal_load_N'][policy].sum(-1).mean()) if policy.any() else None,
            'sensor_coverage':{'independent_positive_samples_per_finger':counts.tolist(),
                'positive_contact_but_zero_taxel_samples_per_finger':misses.tolist(),
                'fraction':np.divide(counts-misses,counts,out=np.ones(5,dtype=float),where=counts>0).tolist(),
                'interpretation':'Presence agreement with independently extracted IPC normal contact; not force calibration.'},
            'wall_seconds':time.monotonic()-started,'complete_planned_trajectory':end_tick==planned_end,
            'model_unchanged':None if not args.model else hashlib.sha256(args.model.read_bytes()).hexdigest()==model_info['sha256']}
        write(out/'score.json',score);write(out/'status.json',{'status':'completed','score':'score.json'})
        print(json.dumps(score),flush=True)
    except BaseException as exc:
        if samples:
            np.savez_compressed(out/'partial_trace.npz',**{k:np.asarray([r[k] for r in samples]) for k in samples[0]})
        write(out/'status.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if scene is not None:scene.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--controller',required=True)
    p.add_argument('--mass-kg',type=float,default=.1002375)
    p.add_argument('--friction',type=float,default=.8)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--fixed-gain',type=float);g.add_argument('--model',type=Path)
    p.add_argument('--substeps',type=int,default=2)
    p.add_argument('--newton-tolerance',type=float,default=.005)
    p.add_argument('--maxvolume',type=float,default=1e-7)
    p.add_argument('--stop-time',type=float)
    p.add_argument('--render',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
