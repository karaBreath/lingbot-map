"""Automatic metric scale from the assumed handheld camera height.

Run:  .venv-amd/Scripts/python.exe -X utf8 tests/test_autoscale.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from lingbot_plus.autoscale import auto_scale_from_camera

    # normal room scan: camera ~0.8 model-units above the floor, scene ~5.5 units.
    # assuming 1.45 m handheld height -> 1.45/0.8 ≈ 1.8125 m per unit.
    scale, note, conf = auto_scale_from_camera(0.8, 5.5, assumed_m=1.45)
    assert abs(scale - 1.45 / 0.8) < 1e-6, f"scale {scale}"
    assert conf is True, "should be confident for a normal floor"
    # a 3-unit wall then measures 3*1.8125 ≈ 5.44 m — plausible room size
    assert 4.0 < 3 * scale < 7.0

    # desk trap: camera sits almost on the plane (tabletop) -> not confident
    s2, n2, c2 = auto_scale_from_camera(0.04, 5.5, assumed_m=1.45)
    assert c2 is False, "tabletop-height plane must be flagged low-confidence"

    # degenerate: zero camera height -> no scale
    s3, n3, c3 = auto_scale_from_camera(0.0, 5.5)
    assert s3 is None and c3 is False

    # scale is inversely proportional to assumed height
    sa, _, _ = auto_scale_from_camera(0.8, 5.5, assumed_m=1.2)
    sb, _, _ = auto_scale_from_camera(0.8, 5.5, assumed_m=1.6)
    assert sb > sa, "taller assumed height -> larger scale"

    print(f"OK — auto-scale {scale:.4f} m/unit (confident), desk flagged, zero handled")


if __name__ == "__main__":
    main()
