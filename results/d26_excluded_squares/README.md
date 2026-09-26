# D26 determinism check

`determinism.py` draws 10,000 `(start, target)` pairs from `T1PickAndPlaceTask._resolve_squares(None, None)`
(default config, `RoboChessEnv().reset(seed=123)`) and prints the count and SHA-256 of `repr(pairs)`.

Run from the repo root: `PYTHONPATH=. python results/d26_excluded_squares/determinism.py`

| Code | Pairs | SHA-256 |
|---|---|---|
| master `a42dfa7`, before any D26 edit | 10,000 | `868586369d198f25602f4a1f2851b75ff878636befc2c659bc3969e5ca24f28e` |
| branch `excluded-squares`, `excluded_squares` at its default (empty) | 10,000 | `868586369d198f25602f4a1f2851b75ff878636befc2c659bc3969e5ca24f28e` |

Identical: with the default, the sampling pool and every RNG draw are unchanged. The "before" hash is pinned in
`tests/test_d26_excluded_squares.py` (`BASELINE_PAIRS_SHA256`).
