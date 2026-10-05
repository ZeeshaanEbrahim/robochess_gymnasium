# RS31 expert gate, 24 training pairs (Section 1.4)

Command (run at 9df7561 in the rs31 worktree, exit 0, 29.0 s wall, max RSS 1,025,496 KB):

    python -W error::UserWarning scripts/gate_expert_cartesian.py --pairs c2:c3,c3:c4,c4:c5,c5:c6,c6:c7,c7:c8,d2:d3,d3:d4,d4:d5,d5:d6,d6:d7,d7:d8,e2:e3,e3:e4,e4:e5,e5:e6,e6:e7,e7:e8,f2:f3,f3:f4,f4:f5,f5:f6,f6:f7,f7:f8 --out gate24.json > gate24.log

Harness v3, budget 300 steps. Criterion (as RS30): final_info.is_success and steps <= 200. clean_place is the harness's separate reporting column, not the criterion.

| pair | is_success | steps | clean_place | terminated_reason |
|---|---|---|---|---|
| c2:c3 | True | 150 | True | success |
| c3:c4 | True | 156 | True | success |
| c4:c5 | True | 147 | True | success |
| c5:c6 | True | 139 | True | success |
| c6:c7 | True | 130 | False | success |
| c7:c8 | True | 142 | False | success |
| d2:d3 | True | 168 | True | success |
| d3:d4 | True | 157 | True | success |
| d4:d5 | True | 148 | False | success |
| d5:d6 | True | 140 | False | success |
| d6:d7 | True | 130 | False | success |
| d7:d8 | True | 142 | False | success |
| e2:e3 | True | 165 | True | success |
| e3:e4 | True | 156 | True | success |
| e4:e5 | True | 146 | False | success |
| e5:e6 | True | 139 | False | success |
| e6:e7 | True | 147 | False | success |
| e7:e8 | True | 143 | False | success |
| f2:f3 | True | 162 | True | success |
| f3:f4 | True | 155 | False | success |
| f4:f5 | True | 145 | False | success |
| f5:f6 | True | 147 | True | success |
| f6:f7 | True | 146 | False | success |
| f7:f8 | True | 143 | False | success |

24 of 24 pairs succeed within 200 steps; steps min 130, max 168.
