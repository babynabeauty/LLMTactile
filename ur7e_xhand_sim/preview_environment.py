"""Render the actual Sandbox and verify the imported platform's resting contact."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from environment import Sandbox


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/platform_preview')
    args=parser.parse_args()
    out=Path(args.output).resolve()
    sim=Sandbox(out, render=False, overview=True)
    for _ in range(100):
        sim.scene.step()
    position=sim.array(sim.cube.get_pos()).reshape(3)
    bottom=float(position[2]-sim.object_spec.bounds_size_m[2]/2)
    assert abs(bottom-.2)<.002, f'Object did not settle on tabletop: {bottom}'
    assert sim.robot.n_dofs==18
    sim.render_enabled=True
    sim.render()
    rgb=sim.array(sim.overview_camera.render(rgb=True)[0]).squeeze().astype('uint8')
    Image.fromarray(rgb).save(out/'overview.png')
    report={'platform':sim.platform_metadata,'robot_dofs':sim.robot.n_dofs,
        'camera_views':sim.camera_views,'object_position_m':position.tolist(),
        'object_bottom_z_m':bottom,'physics_steps':100,
        'description':sim.describe()}
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
    print(json.dumps({'output':str(out),'object_bottom_z_m':bottom}))
    sim.scene.destroy()

if __name__=='__main__':main()
