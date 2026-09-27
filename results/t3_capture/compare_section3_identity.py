"""Compare all episode fields and trace hashes against d28-final runs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
summary={}
for task in ('t1','t2_0','t2_2'):
    old=json.loads((root/f'section3_{task}_d28.json').read_text())['results']
    new=json.loads((root/f'section3_{task}_current.json').read_text())['results']
    assert len(old)==len(new)
    rows=[]
    for baseline,current in zip(old,new):
        assert (baseline['start'],baseline['target'])==(current['start'],current['target'])
        changes={key:dict(baseline=baseline.get(key),current=current.get(key))
                 for key in baseline.keys()|current.keys() if baseline.get(key)!=current.get(key)}
        rows.append(dict(start=current['start'],target=current['target'],changes=changes))
    (root/f'section3_{task}_episode_diff.json').write_text(json.dumps(rows,indent=2))
    failures=[r for r in new if not r['success']]
    self_contacts = [r for r in failures if any(
        {geom['body'] for geom in pair} == {'ur_upper_arm_link','ur_wrist_2_link'}
        for pair in r['contacts'])]
    assert len(self_contacts) == 10
    assert {r['start'] for r in self_contacts} == {'d1','e1'}
    summary[task]=dict(episodes=len(new),successes=sum(r['success'] for r in new),
                       episode_differences=sum(bool(r['changes']) for r in rows),
                       self_contact_failures=len(self_contacts),
                       failures=[{k:r[k] for k in ('start','target','outcome','failed_phase','steps')} for r in failures])
    print(task,json.dumps(summary[task]))
(root/'section3_identity_summary.json').write_text(json.dumps(summary,indent=2))
assert all(v['episode_differences']==0 for v in summary.values())
print('All per-episode fields and full step trace hashes match d28-final')
