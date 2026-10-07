"""POST HOC / EXPLORATORY helpers shared by the diagnosis scripts (imports ../rs32_lib.py)."""
import sys
from pathlib import Path
DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG.parent))
from rs32_lib import *   # EV, REPO, NAMES, SEEDS, RUN, PAIRS, NEIGH, LONG, RS31_PAIRS, wilson, fmt, overlap, monitor, chunk_counts
import numpy as np, json, collections
FAIL_SEEDS = ["rs32D_seed0", "rs32D_seed200", "rs32D_seed500"]
Rk = lambda sq: int(sq[1]); Fl = lambda sq: "abcdefgh".index(sq[0])
def pair_cat(p):
    """neigh_fwd/lat, neigh_back, long_fwd/lat, long_back (back = target rank < start rank, i.e. towards rank 1)."""
    s, t = p.split(":"); back = Rk(t) < Rk(s)
    return ("long_" if p in LONG else "neigh_") + ("back" if back else "fwd/lat")
CATS = ["neigh_fwd/lat", "neigh_back", "long_fwd/lat", "long_back"]
def eval_eps(name):
    return json.load(open(EV / f"{name}_final.json"))["episodes"]
pk = lambda e: f'{e["start_square"]}:{e["target_square"]}'
