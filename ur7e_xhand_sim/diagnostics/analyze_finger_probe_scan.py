"""Summarize controlled scan records and compare sensor-only changes."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/tactile-thumb-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--before',type=Path)
    parser.add_argument('--no-plots',action='store_true',help='Update analysis JSON only; preserve existing figures.')
    args=parser.parse_args()
    results={}
    groups={path.parent.name:read(path) for path in sorted(args.input.glob('*/records.jsonl'))}
    for name,rows in groups.items():
        positive_load_threshold_N=1e-8
        results[name]=dict(samples=len(rows),finite=all(row['all_finite'] for row in rows),
            no_contact_zero_response=all(row['no_contact_zero_response'] for row in rows),
            analytic_mask_matches=sum(row['active_mask_matches_analytic'] for row in rows),
            contact_but_zero_response_samples=sum(row['contact_count']>0 and not row['active_taxel_ids_1based'] for row in rows),
            positive_load_threshold_N=positive_load_threshold_N,
            positive_load_samples=sum(row['total_contact_load_N']>positive_load_threshold_N for row in rows),
            positive_load_but_zero_response_samples=sum(row['total_contact_load_N']>positive_load_threshold_N
                and not row['active_taxel_ids_1based'] for row in rows),
            zero_load_contact_record_and_zero_response_samples=sum(row['contact_count']>0
                and row['total_contact_load_N']<=positive_load_threshold_N
                and not row['active_taxel_ids_1based'] for row in rows),
            sampled_initial_gaps_mm=sorted(set(row['nominal_gap_mm'] for row in rows)))
        if args.before and (args.before/name/'records.jsonl').exists():
            before=read(args.before/name/'records.jsonl')
            index=lambda row:(row['finger'],row['shape'],row['target'],row['nominal_gap_mm'],row['steps_after_initial_placement'])
            before={index(row):row for row in before}
            pose_errors=[];force_errors=[];contact_equal=[];changed_masks=0
            for row in rows:
                old=before[index(row)]
                pose_errors.append(np.max(abs(np.r_[row['after_position_m'],row['after_quaternion_wxyz']]-
                                                np.r_[old['after_position_m'],old['after_quaternion_wxyz']])))
                force_errors.append(abs(row['total_contact_load_N']-old['total_contact_load_N']))
                contact_equal.append(row['contacts']==old['contacts'])
                changed_masks+=row['active_taxel_ids_1based']!=old['active_taxel_ids_1based']
            results[name]['paired_comparison']=dict(max_pose_component_error=float(max(pose_errors)),
                max_total_contact_force_error_N=float(max(force_errors)),
                all_contact_records_exactly_equal=all(contact_equal),changed_tactile_masks=int(changed_masks))
    (args.input/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    if args.no_plots:
        print(json.dumps(results,indent=2))
        return
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors={'central':'#1971c2','edge':'#2b8a3e','tip':'#e67700','source_contact_tip':'#c92a2a'}
    for ax,name in zip(axes.ravel(),('thumb_sphere','thumb_plate','index_sphere','index_plate')):
        if name not in groups:
            ax.axis('off');continue
        rows=groups[name]
        for target in dict.fromkeys(row['target'] for row in rows):
            data=sorted((row for row in rows if row['target']==target and row['steps_after_initial_placement']==1),key=lambda r:r['nominal_gap_mm'])
            x=[r['nominal_gap_mm'] for r in data]
            measured=[len(r['active_taxel_ids_1based']) for r in data]
            expected=[len(r['expected_active_taxel_ids_1based']) for r in data]
            ax.plot(x,expected,':',color=colors[target],alpha=.6)
            ax.plot(x,measured,'o-',color=colors[target],label=target,markersize=4)
        ax.set(title=name.replace('_',' / '),xlabel='Nominal initial surface gap (mm)',ylabel='Responding taxels / 120')
        ax.invert_xaxis();ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Controlled initial placement: active tactile points after one physics step\nSolid = recorded; dotted = analytic overlap plus physical-contact gate; this is not force calibration',fontsize=12)
    fig.savefig(args.input/'controlled_scan_active_points.png',dpi=180)
    plt.close(fig)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
