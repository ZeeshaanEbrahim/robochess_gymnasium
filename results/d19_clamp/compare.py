"""Build the D19 comparison table from the eval JSONs and saturation probe outputs."""
import ast, json, re
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
D = Path(__file__).parent; U = D.parent / "delta_sanity"
J = ["pan", "lift", "elbow", "w1", "w2", "w3"]
def sat(p):
    t = p.read_text()
    s = ast.literal_eval(re.search(r"SATURATED:\s+(\{.*\})", t).group(1))
    e = ast.literal_eval(re.search(r"actual\| deg, median:\s+(\{.*\})", t).group(1))
    return "/".join(f"{s[j]:.2f}" for j in J), "/".join(f"{e[j]:.0f}" for j in J)
def ep_min(d): return np.array([e["min_gripper_to_piece_m"] for e in d["episodes"]])
rows, eps = [], {}
for label, ev, sp in [("unclamped delta", U / "ck{}.json", D / "saturation_unclamped_ck{}.txt"),
                      ("clamped delta", D / "clamp_ck{}.json", D / "saturation_clamp_ck{}.txt")]:
    for ck in (20000, 40000, 60000, 80000, 100000):
        d = json.loads(Path(str(ev).format(ck)).read_text()); m = d["min_gripper_to_piece_m"]
        s, e = sat(Path(str(sp).format(ck)))
        rows.append((label, ck // 1000, m["median"], m["min"], int((ep_min(d) < 0.10).sum()), d["grasp_ever_count"], s, e))
        eps[(label, ck)] = ep_min(d)
d = json.loads((D / "absolute_pilot_ck100000.json").read_text()); m = d["min_gripper_to_piece_m"]
s, e = sat(U / "saturation_absolute_pilot_ck100000.txt")
rows.append(("absolute pilot", 100, m["median"], m["min"], int((ep_min(d) < 0.10).sum()), d["grasp_ever_count"], s, e))
print("| run | ckpt | median min dist (m) | closest (m) | eps <10cm | real grasps | saturated pan/lift/elbow/w1/w2/w3 | median cmd-actual deg |")
print("|---|---|---|---|---|---|---|---|")
for r in rows: print(f"| {r[0]} | {r[1]}k | {r[2]:.3f} | {r[3]:.3f} | {r[4]}/50 | {r[5]} | {r[6]} | {r[7]} |")
c20, c100, u100 = eps[("clamped delta", 20000)], eps[("clamped delta", 100000)], eps[("unclamped delta", 100000)]
print(f"\nclamped 20k->100k paired: median diff {np.median(c100 - c20):+.3f} m, closer in {(c100 < c20).sum()}/50, Wilcoxon p={wilcoxon(c20, c100).pvalue:.2g}")
print(f"clamped vs unclamped @100k paired: median diff {np.median(c100 - u100):+.3f} m, clamped closer in {(c100 < u100).sum()}/50, Wilcoxon p={wilcoxon(c100, u100).pvalue:.2g}")
