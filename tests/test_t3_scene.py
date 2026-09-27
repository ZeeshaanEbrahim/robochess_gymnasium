"""D13: deterministic generation retains every original scene byte."""
from scripts.build_t3_scene import (
    CAPTURED_QPOS, GENERATED, HEADER, PAD_BLOCK, SOURCE, build, captured_body,
)


def test_t3_scene_regenerates_byte_identically():
    assert build(SOURCE.read_bytes().decode()).encode() == GENERATED.read_bytes()


def test_t3_scene_additions_leave_source_byte_identical():
    source = SOURCE.read_bytes()
    generated = GENERATED.read_bytes().decode()
    assert generated.startswith(HEADER)
    original = generated[len(HEADER):]
    for block in (PAD_BLOCK, captured_body(source.decode()), CAPTURED_QPOS):
        assert original.count(block) == 1
        original = original.replace(block, "", 1)
    assert original.encode() == source
