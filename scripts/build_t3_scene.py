#!/usr/bin/env python3
"""Generate scene_t3.xml by unique text insertions, following the T2 builder."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SOURCE = ASSETS / "scene.xml"
GENERATED = ASSETS / "scene_t3.xml"
HEADER = """<!-- GENERATED FILE -- do not edit by hand.
     Produced by scripts/build_t3_scene.py from scene.xml (D13).
     T1 and T2 keep their original scenes and physics. -->
"""
BOARD_ANCHOR = '    <body name="board" pos="0 0.52 0.7560">\n'
WORLD_ANCHOR = "    </body>\n  </worldbody>"
KEY_ANCHOR = '  0.0390 0.4810 0.7620 1 0 0 0"'
CAPTURED_QPOS = "  0.0390 0.5590 0.7620 1 0 0 0"
PAD_BLOCK = """      <!-- T3 capture pad: collision belongs to board for F1 and held. -->
      <geom name="capture_pad" type="box" size="0.0390 0.0390 0.0060"
        pos="0.3510 -0.1950 0" material="board_light" friction="0.7 0.02 0.001"/>
"""


def replace_once(text: str, anchor: str, replacement: str) -> str:
    if text.count(anchor) != 1:
        raise ValueError(f"scene.xml must contain exactly one anchor: {anchor!r}")
    return text.replace(anchor, replacement, 1)


def captured_body(source_text: str) -> str:
    bodies = re.findall(
        r'    <body name="piece_a" pos="0.0390 0.4810 0.7620">.*?\n    </body>\n',
        source_text, re.S,
    )
    if len(bodies) != 1:
        raise ValueError("scene.xml must contain exactly one expected piece_a body.")
    return replace_once(
        bodies[0].replace("piece_a", "piece_b"),
        'pos="0.0390 0.4810 0.7620"', 'pos="0.0390 0.5590 0.7620"',
    )


def build(source_text: str) -> str:
    # D13/T2 pattern: retain the original bytes, including MuJoCo's permissive
    # comments. If a source anchor changes, fail rather than silently drift.
    text = replace_once(source_text, BOARD_ANCHOR, BOARD_ANCHOR + PAD_BLOCK)
    text = replace_once(text, WORLD_ANCHOR,
                        "    </body>\n" + captured_body(source_text) + "  </worldbody>")
    text = replace_once(text, KEY_ANCHOR, KEY_ANCHOR[:-1] + CAPTURED_QPOS + '"')
    return HEADER + text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = build(SOURCE.read_bytes().decode()).encode()
    if args.check:
        if not GENERATED.exists() or GENERATED.read_bytes() != generated:
            print("assets/scene_t3.xml is out of date; run scripts/build_t3_scene.py")
            return 1
        print("assets/scene_t3.xml is byte-identical to regeneration")
        return 0
    GENERATED.write_bytes(generated)
    print(f"wrote {GENERATED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
