"""Unit tests for room segmentation robustness to fragmented walls/floor.

Two real failure modes on drift-fragmented reconstructions:
  A) patchy floor + stray in-room wall specks shatter ONE room into zero/many
  B) a dividing wall broken into dashes stops dividing -> two rooms merge to one

segment_rooms() must: bridge floor gaps, drop isolated wall specks, and bridge
small gaps in genuine (long) walls so they keep dividing.

Run:  .venv-amd/Scripts/python.exe -X utf8 tests/test_rooms.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fragmented_single_room():
    """One room, floor punched full of holes + random wall specks inside.
    Expect exactly 1 room recovered."""
    from lingbot_plus.floorplan import segment_rooms

    rng = np.random.default_rng(0)
    H, W = 130, 110
    floor = np.zeros((H, W), bool)
    floor[20:110, 20:90] = True
    # punch ~45% holes (fragmentation)
    holes = rng.random((H, W)) < 0.45
    floor &= ~holes
    # stray in-room wall specks (furniture edges misread as wall)
    wall = np.zeros((H, W), bool)
    for _ in range(40):
        y, x = rng.integers(25, 105), rng.integers(25, 85)
        wall[y, x] = True

    labels, rooms = segment_rooms(
        floor, wall, per_cell=0.04, scaled=True, min_room_m2=1.5,
        min_cells=400)
    assert len(rooms) == 1, f"fragmented single room -> {len(rooms)} rooms (want 1)"
    # area ballpark: 90*70 cells * 0.04^2, minus holes/erosion — accept 60-160%
    a = rooms[0]["area"]
    assert 6.0 < a < 16.0, f"room area {a:.1f} m² off (nominal ~10)"
    print(f"  A) fragmented single room -> 1 room, area {a:.1f} m² OK")


def test_broken_dividing_wall():
    """Two rooms split by a wall broken into dashes. Expect 2 rooms."""
    from lingbot_plus.floorplan import segment_rooms

    H, W = 100, 140
    floor = np.zeros((H, W), bool)
    floor[15:85, 15:125] = True
    wall = np.zeros((H, W), bool)
    # vertical divider at x=70, dashed (2-cell gaps every 6 cells)
    for y in range(15, 85):
        if (y // 3) % 2 == 0:
            wall[y, 69:71] = True
    # some enclosing wall too (helps neither side leak around the top/bottom)
    wall[14:86, 14] = True
    wall[14:86, 125] = True
    wall[14, 14:126] = True
    wall[85, 14:126] = True

    labels, rooms = segment_rooms(
        floor, wall, per_cell=0.04, scaled=True, min_room_m2=1.5,
        min_cells=400)
    assert len(rooms) == 2, f"broken dividing wall -> {len(rooms)} rooms (want 2)"
    print(f"  B) broken dividing wall -> 2 rooms OK "
          f"({rooms[0]['area']:.1f} + {rooms[1]['area']:.1f} m²)")


def main():
    test_fragmented_single_room()
    test_broken_dividing_wall()
    print("OK — room segmentation robust to fragmented walls/floor")


if __name__ == "__main__":
    main()
