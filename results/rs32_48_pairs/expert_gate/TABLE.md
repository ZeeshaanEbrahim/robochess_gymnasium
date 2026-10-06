# RS32 expert gate, 48 training pairs + 6 reserves (Section 1.5)

Harness v3, budget 300 steps, `scripts/gate_expert_cartesian.py` (same as RS31). Command, run in the rs32 worktree at 0a53727, exit 0, 30.63 s wall, max RSS 1,305,640 KB:

    python -W error::UserWarning scripts/gate_expert_cartesian.py --pairs <54 pairs, comma separated> --out gate54.json > gate54.log

Criterion (as RS30/RS31): final is_success and steps <= 200. clean_place is reported, not the criterion.

| pair | group | is_success | steps | clean_place | verdict |
|---|---|---|---|---|---|
| c2:c3 | neighbour | True | 150 | True | PASS |
| c3:c4 | neighbour | True | 156 | True | PASS |
| c4:c5 | neighbour | True | 147 | True | PASS |
| c5:c6 | neighbour | True | 139 | True | PASS |
| c6:c7 | neighbour | True | 130 | False | PASS |
| c7:c8 | neighbour | True | 142 | False | PASS |
| d2:d3 | neighbour | True | 168 | True | PASS |
| d3:d4 | neighbour | True | 157 | True | PASS |
| d4:d5 | neighbour | True | 148 | False | PASS |
| d5:d6 | neighbour | True | 140 | False | PASS |
| d6:d7 | neighbour | True | 130 | False | PASS |
| d7:d8 | neighbour | True | 142 | False | PASS |
| e2:e3 | neighbour | True | 165 | True | PASS |
| e3:e4 | neighbour | True | 156 | True | PASS |
| e4:e5 | neighbour | True | 146 | False | PASS |
| e5:e6 | neighbour | True | 139 | False | PASS |
| e6:e7 | neighbour | True | 147 | False | PASS |
| e7:e8 | neighbour | True | 143 | False | PASS |
| f2:f3 | neighbour | True | 162 | True | PASS |
| f3:f4 | neighbour | True | 155 | False | PASS |
| f4:f5 | neighbour | True | 145 | False | PASS |
| f5:f6 | neighbour | True | 147 | True | PASS |
| f6:f7 | neighbour | True | 146 | False | PASS |
| f7:f8 | neighbour | True | 143 | False | PASS |
| c4:d4 | neighbour | True | 156 | False | PASS |
| f5:e5 | neighbour | True | 156 | True | PASS |
| d3:e4 | neighbour | True | 157 | True | PASS |
| e6:d7 | neighbour | True | 159 | False | PASS |
| e5:e4 | neighbour | True | 145 | False | PASS |
| c6:c5 | neighbour | True | 136 | False | PASS |
| b2:b4 | longer | True | 162 | False | PASS |
| g3:g6 | longer | True | 167 | False | PASS |
| d7:d4 | longer | True | 160 | False | PASS |
| a5:c5 | longer | True | 143 | False | PASS |
| h4:e4 | longer | True | 173 | False | PASS |
| b6:e6 | longer | True | 159 | False | PASS |
| a3:c5 | longer | True | 169 | False | PASS |
| c3:f6 | longer | True | 172 | False | PASS |
| h3:f5 | longer | True | 175 | False | PASS |
| g7:d4 | longer | True | 171 | False | PASS |
| b7:d5 | longer | True | 159 | False | PASS |
| e2:g4 | longer | True | 173 | False | PASS |
| a3:d5 | longer | True | 176 | False | PASS |
| h6:e4 | longer | True | 167 | False | PASS |
| c2:e5 | longer | True | 174 | False | PASS |
| f3:d6 | longer | True | 180 | False | PASS |
| b4:e6 | longer | True | 169 | False | PASS |
| g6:e3 | longer | True | 167 | False | PASS |
| c5:d6 | reserve | True | 145 | False | PASS |
| f4:e3 | reserve | True | 153 | False | PASS |
| d4:c4 | reserve | True | 152 | False | PASS |
| b3:b5 | reserve | True | 155 | False | PASS |
| g5:e5 | reserve | True | 160 | False | PASS |
| c6:f4 | reserve | True | 149 | False | PASS |

Failures among the 48: 0 []. Reserves needed: none.
Steps over the 48: min 130, max 180. Reserves: min 145, max 160.
