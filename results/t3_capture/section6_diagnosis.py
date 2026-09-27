"""Section 6 measurements only; reuse the unchanged section 4 controller."""
import json
from pathlib import Path
import sys
import statistics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.check_t3_expert import (Observer, PieceView, ScriptedExpert, IKFailure,
    COMPENSATED_WAYPOINTS, make_t3_env, run_pair, contacts, referee_pairs, np)

OUT = Path(__file__).resolve().parent

def removal(square):
    start = 'b8' if square == 'a8' else 'a8'
    env = make_t3_env(start_square=start, target_square=square,
                      action_mode='absolute', max_episode_steps=1000)
    _, info = env.reset(seed=0)
    observer = Observer(env)
    expert = ScriptedExpert(observer)
    expert.base = PieceView(env)
    observer.begin_stage('remove')
    previous_target = previous_q = None
    exception = None
    try:
        for label, pinch, grip, budget in expert.waypoints(
                env._board_world_xy(square), env.zone_position[:2]):
            observer.phase = 'remove:' + label
            reuse = previous_target is not None and np.allclose(pinch, previous_target)
            try:
                q = previous_q if reuse else expert.solve_ik(pinch)
                wp = expert._run_waypoint(pinch, grip, budget, label, None, q,
                    compensate=label in COMPENSATED_WAYPOINTS and not reuse)
            except IKFailure as exc:
                exception = str(exc)
                break
            previous_target, previous_q = pinch.copy(), wp['q_arm']
            info = wp['info']
            if wp['terminated'] or wp['truncated']:
                break
        b = env.unwrapped
        xy = info['captured_release_position']
        edges = None if xy is None else dict(
            left=(xy[0]-env._zone_low[0])*1000,
            right=(env._zone_high[0]-xy[0])*1000,
            near=(xy[1]-env._zone_low[1])*1000,
            far=(env._zone_high[1]-xy[1])*1000)
        result = dict(square=square, parked=start,
            stage_1_reached=observer.stage_1_step is not None,
            stage_1_step=observer.stage_1_step, total_length=b._elapsed_steps,
            phase_lengths=dict(observer.lengths), final_phase=observer.phase,
            release_height_mm=info['captured_release_height_mm'],
            released=info['captured_terms']['released'],
            release_xy_m=xy, pad_edge_distances_mm=edges,
            nearest_pad_edge_mm=None if edges is None else min(edges.values()),
            interactions=observer.metrics['remove'],
            terminated_reason=info['terminated_reason'],
            off_board_piece=info['off_board_piece'],
            exempt_steps=observer.exempt_steps,
            final_carried_xy_m=b.data.xpos[env._captured_bid,:2].tolist())
        if not result['stage_1_reached']:
            result['failure'] = dict(phase=observer.phase,
                terminated_reason=info['terminated_reason'], exception=exception,
                referee_contact_pairs=referee_pairs(b), active_contacts=contacts(b),
                carried_xy_m=result['final_carried_xy_m'])
        return result
    finally:
        env.close()

def main():
    squares = [f+r for r in '12345678' for f in 'abcdefgh'
               if f+r not in ('d1','e1')]
    rows=[]
    for square in squares:
        row=removal(square)
        rows.append(row)
        print('REMOVAL',json.dumps(row),flush=True)
        (OUT/'section6_removal.json').write_text(json.dumps(rows,indent=2))
    pairs=[('h2','a8'),('g1','b8'),('a1','a8'),('h4','a7'),('f1','a8')]
    full=[]
    for start,target in pairs:
        row=run_pair(start,target)
        full.append(row)
        print('CAPTURE',json.dumps(row),flush=True)
        (OUT/'section6_captures.json').write_text(json.dumps(full,indent=2))
    old=json.loads((OUT/'section4_expert.json').read_text())['results']
    successes=[r for r in old+full if r['success']]
    longest=max(successes,key=lambda r:r['total_length'])
    summary=dict(removable=sum(r['stage_1_reached'] for r in rows),
        removal_episodes=len(rows), failing_squares=[r['square'] for r in rows if not r['stage_1_reached']],
        extreme_successes=sum(r['success'] for r in full),
        combined_successes=len(successes), combined_episodes=len(old+full),
        longest_pair=longest['start']+'→'+longest['target'],longest_length=longest['total_length'],
        fit_budgets={str(n):sum(r['total_length']<=n for r in successes) for n in (400,500,600)})
    (OUT/'section6_summary.json').write_text(json.dumps(summary,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()
