#!/usr/bin/env python
"""Builds DIAGNOSIS.md from DIAGNOSIS_template.md by pasting the three output files verbatim at the {{file}} markers."""
from pathlib import Path
D = Path(__file__).resolve().parent
t = (D / "DIAGNOSIS_template.md").read_text()
for f in ("diag_eval_tables_output.txt", "diag_rollout_tables_output.txt", "diag_training_tables_output.txt"):
    assert "{{" + f + "}}" in t, f
    t = t.replace("{{" + f + "}}", (D / f).read_text().rstrip())
assert "{{" not in t
(D / "DIAGNOSIS.md").write_text(t)
print(len(t.splitlines()), "lines")
