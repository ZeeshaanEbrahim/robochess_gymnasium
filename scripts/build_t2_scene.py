#!/usr/bin/env python3
"""Generate assets/scene_t2.xml from assets/scene.xml by adding T2's obstacle pool.

T2 needs obstacle bodies; MuJoCo cannot add bodies after compile, so they have
to exist in the model file. They are NOT added to scene.xml, because adding
free joints to the shared scene changes the constraint solve for every task
using it: measured, adding four obstacle bodies parked 1.2 m away and never
touched moved T1's five-target sweep from 309/315 to 306/315, with 149 of 315
episodes differing numerically. That is the same sensitivity D11 records. T1's
validated figures have to keep reproducing, so T2 gets its own scene.

Generated rather than hand-maintained so the two files cannot drift: this
script is the only thing that writes scene_t2.xml, and
tests/test_t2_obstacle_pick_and_place.py asserts regenerating reproduces the
committed file byte-for-byte.

Usage:
    python scripts/build_t2_scene.py            # write assets/scene_t2.xml
    python scripts/build_t2_scene.py --check    # exit 1 if it is out of date
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SOURCE = ASSETS / "scene.xml"
GENERATED = ASSETS / "scene_t2.xml"

N_OBSTACLES = 4
# Parked on the floor, clear of the table. Kept in step with PARKED_XY in
# robochess/tasks/t2_obstacle_pick_and_place.py, which the test suite checks.
PARKED = tuple((1.20 + 0.20 * i, -1.20) for i in range(N_OBSTACLES))

HEADER = """<!-- GENERATED FILE -- do not edit by hand.
     Produced by scripts/build_t2_scene.py from scene.xml; see that script for
     why T2 has its own scene instead of sharing T1's. Regenerate after any
     change to scene.xml. -->
"""


def obstacle_body(i: int) -> str:
    x, y = PARKED[i]
    n = i + 1
    return f"""    <!-- T2 obstacle slot {n}. Parked on the floor at ({x:.2f}, {y:.2f}) when the
         episode does not use it; T2ObstacleTask teleports it onto a square when
         it does. Identical geometry to piece_a. -->
    <body name="obstacle_{n}" pos="{x:.2f} {y:.2f} 0">
      <freejoint name="obstacle_{n}_free"/>
      <geom class="piece_collision" name="obstacle_{n}_base" type="cylinder" size="0.015 0.009" pos="0 0 0.009"/>
      <geom class="piece_collision" name="obstacle_{n}_stem" type="cylinder" size="0.008 0.008" pos="0 0 0.026"/>
      <geom class="piece_collision" name="obstacle_{n}_head" type="capsule" size="0.009 0.001" pos="0 0 0.040"/>
      <geom class="piece_visual" type="cylinder" size="0.015 0.009" pos="0 0 0.009"/>
      <geom class="piece_visual" type="cylinder" size="0.008 0.008" pos="0 0 0.026"/>
      <geom class="piece_visual" type="capsule" size="0.009 0.001" pos="0 0 0.040"/>
    </body>
"""


def build(source_text: str) -> str:
    marker = "    </body>\n  </worldbody>"
    if source_text.count(marker) != 1:
        raise SystemExit(
            "scene.xml no longer ends its worldbody the way this generator expects; "
            "update scripts/build_t2_scene.py rather than editing the generated file."
        )
    bodies = "\n".join(obstacle_body(i) for i in range(N_OBSTACLES))
    text = source_text.replace(marker, "    </body>\n\n" + bodies + "  </worldbody>")

    # The home keyframe's qpos is positional and must cover every joint, so the
    # parked obstacle poses are appended. The tracked piece keeps its address
    # because the obstacle bodies are appended after it.
    key = '  0.0390 0.4810 0.7620 1 0 0 0"'
    if text.count(key) != 1:
        raise SystemExit("scene.xml's home keyframe is not in the expected form.")
    parked = "  ".join(f"{x:.2f} {y:.2f} 0 1 0 0 0" for x, y in PARKED)
    text = text.replace(key, f'  0.0390 0.4810 0.7620 1 0 0 0  {parked}"')

    return HEADER + text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the committed file matches, do not write.")
    args = parser.parse_args()

    generated = build(SOURCE.read_text())
    if args.check:
        current = GENERATED.read_text() if GENERATED.exists() else None
        if current != generated:
            print("assets/scene_t2.xml is out of date; run scripts/build_t2_scene.py")
            return 1
        print("assets/scene_t2.xml is up to date")
        return 0
    GENERATED.write_text(generated)
    print(f"wrote {GENERATED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
