"""Render the actual Sandbox and verify the imported platform's resting contact."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from environment import Sandbox


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/platform_preview')
    parser.add_argument('--task', choices=('grasp','tube_insert','usb_insert'), default='grasp')
    parser.add_argument('--object-spec-json', type=json.loads)
    parser.add_argument('--settle-steps', type=int)
    args=parser.parse_args()
    if args.settle_steps is not None and args.settle_steps < 1:
        parser.error('--settle-steps must be positive')
    out=Path(args.output).resolve()
    timing = dict(physics_dt_s=1/300, control_substeps=5, noslip_iterations=5, deterministic=True) if args.task != 'grasp' else {}
    sim=Sandbox(out, render=False, overview=True, task=args.task, object_spec=args.object_spec_json, **timing)
    steps=args.settle_steps or (300 if sim.insertion else 100)
    stable_samples=[]
    for i in range(steps):
        sim.scene.step()
        if sim.insertion and i >= steps-60 and (i+1)%5 == 0:
            stable_samples.append((sim.values(sim.cube.get_pos()),sim.values(sim.cube.get_quat())))
    position=sim.array(sim.cube.get_pos()).reshape(3)
    quat=sim.array(sim.cube.get_quat()).reshape(4)
    bottom=float(position[2]-sim.object_spec.vertical_extent(quat))
    assert abs(bottom-.2)<.003, f'Object did not settle on its support: {bottom}'
    stability = None
    if sim.insertion:
        assert len(stable_samples) >= 8, 'Increase --settle-steps to at least 60 to verify settling'
        positions=np.asarray([s[0] for s in stable_samples])
        rotations=Rotation.from_quat(np.asarray([s[1] for s in stable_samples])[:,[1,2,3,0]])
        movement=float(np.linalg.norm(positions-positions[0],axis=1).max())
        angle=float(np.rad2deg((rotations[0].inv()*rotations).magnitude()).max())
        stability=dict(window_s=(len(stable_samples)-1)*5*sim.physics_dt_s,
                       max_translation_from_window_start_m=movement,
                       max_rotation_from_window_start_deg=angle,
                       translation_limit_m=.0005,rotation_limit_deg=1.,
                       passed=bool(movement < .0005 and angle < 1.))
        assert stability['passed'], f'Object is not yet stable: {stability}'
    assert sim.robot.n_dofs==18
    sim.render_enabled=True
    sim.render()
    rgb=sim.array(sim.overview_camera.render(rgb=True)[0]).squeeze().astype('uint8')
    Image.fromarray(rgb).save(out/'overview.png')
    if sim.insertion:
        rgb=sim.array(sim.task_camera.render(rgb=True)[0]).squeeze().astype('uint8')
        Image.fromarray(rgb).save(out/'task_rgb.png')
    report={'platform':sim.platform_metadata,'robot_dofs':sim.robot.n_dofs,
        'camera_views':sim.camera_views,'object_position_m':position.tolist(),
        'object_bottom_z_m':bottom,'physics_steps':steps,
        'settling_check':stability,
        'task':args.task,'evaluation':sim.evaluate(),
        'description':sim.describe()}
    if sim.insertion:
        from validate_insertion_collision import validate_collision_geometry
        report['insertion_collision_check']=validate_collision_geometry(sim)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    sheet=Image.new('RGB',(1600,1200),'#121922')
    draw=ImageDraw.Draw(sheet)
    for name,label,box in [('overview','PLATFORM / SIMULATION OVERVIEW',(0,0,1000,700)),
        ('scene_front','FRONT CAMERA',(1000,0,1600,400)),
        ('scene_left','LEFT CAMERA',(1000,400,1600,800)),
        ('wrist','WRIST CAMERA',(0,700,800,1200)),
        ('hand_thumb','HAND DETAIL',(800,800,1600,1200))]:
        x,y,x1,y1=box
        im=Image.open(out/f'{name}.png');im.thumbnail((x1-x-16,y1-y-40))
        sheet.paste(im,(x+(x1-x-im.width)//2,y+32))
        draw.text((x+14,y+10),label,fill='white')
    sheet.save(out/'contact_sheet.png')
    if sim.insertion:
        comparison=Image.new('RGB',(1800,900),'#121922')
        overview=Image.open(out/'overview.png').resize((900,675))
        detail=Image.open(out/'task_rgb.png').resize((900,675))
        comparison.paste(overview,(0,100))
        comparison.paste(detail,(900,100))
        draw=ImageDraw.Draw(comparison)
        draw.text((24,48),'UR7e + XHAND1 / '+args.task,fill='white')
        draw.text((924,48),'TASK ASSETS / PHYSICS SETTLED / ORIGINAL SCALE',fill='white')
        comparison.save(out/'scene_preview.png')
    if sim.insertion:
        assert report['insertion_collision_check']['pass'], f'Insertion collision validation failed; see {out}/validation.json'
    print(json.dumps({'output':str(out),'object_bottom_z_m':bottom}))
    sim.scene.destroy()

if __name__=='__main__':main()
