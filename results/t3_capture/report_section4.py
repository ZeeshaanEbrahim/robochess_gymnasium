"""Format saved measurements; does not run or modify any simulation."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
x=json.loads((root/'section4_expert.json').read_text())
results=x['results'];summary=x['summary']
phase_order=('approach','descend','close','lift','traverse','place','release','retreat')
def number(value,places=6):return '—' if value is None else f'{value:.{places}f}'
def pair(r):return r['start']+'→'+r['target']
def contact_pairs(r):
    return sorted({tuple(c['pair']) for c in r.get('failure',{}).get('active_contacts',[])})
lines=['# T3 section 4 measurements','',
       'Command: `nice -n 19 python scripts/check_t3_expert.py` — exit **0**.',
       'All episodes use `max_episode_steps=1000`, joint absolute mode and seed 0.',
       'The expert source, constants and T3 executable rules were unchanged.','',
       '## Measurement definitions','',
       '- `Stage-1 step` is the wrapper flag latch. `Stage-2 length` is total minus that latch, including the remaining removal release/retreat.',
       '- Interaction stages follow the controller sequences: **remove** measures gripper versus capturing piece_a; **move** measures gripper versus captured piece_b.',
       '- Signed geom distances include all gripper geoms (visual and collision) and all geoms on the other piece. Measurements are at reset, controller handoff and macro-step boundaries, not every physics substep.',
       '- Peak displacement is the 3D norm relative to that other piece’s position at the start of the controller sequence. Tilt is absolute tilt from upright.',
       '- Pad-edge distances are signed centre-to-edge distances at the last bilateral contact; edge order is left, right, near (low y), far (high y). Negative means outside.',
       '- D27 release records are last-bilateral-contact records. For a failure while still held, they are labelled **unreleased**, not reported as an actual release event.','',
       '## Summary','',f"Successes: **{summary['successes']}/{summary['episodes']}**.",
       f"Successful total lengths: min **{summary['successful_lengths']['min']}**, median **{summary['successful_lengths']['median']}**, max **{summary['successful_lengths']['max']}**.",'',
       '| Step cap | Successes fitting |','|---|---:|']
for budget,count in summary['fit_budgets'].items():lines.append(f'| {budget} | {count} |')
lines+=['','## Pair outcomes','','| Pair | Success | Stage-1 step | Stage-2 length | Total |','|---|---|---:|---:|---:|']
for r in results:
    lines.append(f"| {pair(r)} | {'yes' if r['success'] else 'no'} | {r['stage_1_step'] if r['stage_1_step'] is not None else '—'} | {r['stage_2_length'] if r['stage_2_length'] is not None else '—'} | {r['total_length']} |")
for r in results:
    cap=r['captured_release'];actor=r['capturing_release']
    lines+=['',f'## {pair(r)}','',f"Success: **{r['success']}**; stage-1 step: **{r['stage_1_step']}**; stage-2 length: **{r['stage_2_length']}**; total: **{r['total_length']}**.",'',
            '### Phase lengths','','| Controller | '+' | '.join(phase_order)+' |','|---|'+'---:|'*len(phase_order)]
    for stage in ('remove','move'):
        values=[str(r['phase_lengths'].get(stage+':'+p,'—')) for p in phase_order]
        lines.append('| '+stage+' | '+' | '.join(values)+' |')
    lines+=['','### Release records','','| Measurement | Value |','|---|---|',
            f"| Captured height above pad (mm) | {number(cap['height_mm'])} {'(unreleased)' if not cap['released'] and cap['height_mm'] is not None else ''} |",
            f"| Captured XY world position (mm) | {', '.join(number(v*1000) for v in cap['xy_m']) if cap['xy_m'] is not None else '—'} |",
            f"| Captured pad-edge distances (mm), left/right/near/far | {', '.join(number(v) for v in cap['pad_edge_distances_mm'].values()) if cap['pad_edge_distances_mm'] else '—'} |",
            f"| Capturing height above board (mm), D27 record | {number(actor['height_mm'])} {'(unreleased)' if not r['success'] and actor['height_mm'] is not None else ''} |",
            f"| Capturing XY error (mm), D27 record | {number(actor['xy_error_mm'])} |",
            '', '### Other-piece interactions','',
            '| Controller | Other piece | Minimum signed geom distance (mm) | Peak displacement (mm) | Peak tilt (degrees) | Closest geom pair | Step |',
            '|---|---|---:|---:|---:|---|---:|']
    for stage in ('remove','move'):
        metric=r['interactions'].get(stage)
        if metric:
            lines.append(f"| {stage} | {metric['other_piece']} | {number(metric['min_distance_mm'])} | {number(metric['peak_displacement_mm'],9)} | {number(metric['peak_tilt_deg'],9)} | {' ↔ '.join(metric['min_pair'])} | {metric['min_step']} |")
        else:lines.append(f'| {stage} | not reached | — | — | — | — | — |')
    if not r['success']:
        failure=r['failure']
        lines+=['','### Failure','',f"- Phase: `{failure['phase']}`; `terminated_reason={failure['terminated_reason']}`; off-board piece: `{r['off_board_piece']}`.",
                f"- Referee contact pair: {failure['referee_contact_pairs'] or 'none: off-board boundary termination, not illegal contact'}.",
                '- Active contact pairs (legal grasp/support contacts):']
        lines += ['  - `'+' ↔ '.join(p)+'`' for p in contact_pairs(r)]
lines+=['','## Diagnosis','',
        '- g6→h8 and h1→h8 fail at the same removal traverse step. Captured piece_b is still pinched, at x=0.355328 m and y=0.603009 m: beyond the board’s side envelope and outside the pad footprint. Removal/place/release is never reached.',
        '- h8→h1 completes removal, then capturing piece_a crosses the board boundary during its joint-space traverse while still pinched.',
        '- h1→h8 fails before the capturing-piece traverse. It therefore does not reproduce that original D10 traversal in this episode.',
        '- No fixes were made to the expert, constants, physics or T3 rules.']
(root/'section4_report.md').write_text('\n'.join(lines)+'\n')
print('SUMMARY',json.dumps(summary))
for r in results:
    print('PAIR',pair(r),r['success'],r['stage_1_step'],r['stage_2_length'],r['total_length'])
for stage in ('remove','move'):
    metrics=[r['interactions'][stage] for r in results if stage in r['interactions']]
    print('INTERACTIONS',stage,dict(min_distance_mm=round(min(m['min_distance_mm'] for m in metrics),6),
                                  max_peak_displacement_mm=round(max(m['peak_displacement_mm'] for m in metrics),9),
                                  max_peak_tilt_deg=round(max(m['peak_tilt_deg'] for m in metrics),9)))
print('Gripper geom counts:',sorted({r['gripper_geom_count'] for r in results}))
print('Order violations:',sum(r['order_violation'] for r in results))
print('Report:',root/'section4_report.md')
