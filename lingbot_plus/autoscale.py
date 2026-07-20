"""Automatic metric scale for a monocular scan (no manual measurement).

Monocular reconstruction has no inherent scale. The one physical reference
always present in a handheld room scan is the camera's height above the floor —
people hold a phone at roughly chest/eye level while walking. Assuming that
height turns model units into metres, accurate to about ±20% (real height and
grip vary). This is a ballpark, clearly labelled as such; the manual measure
tool (R1) stays authoritative when the user wants exact numbers.

Guards the desk trap: if the detected "floor" sits implausibly close to the
cameras (a tabletop, not the real floor), the estimate is flagged low-confidence
instead of silently producing a wrong scale.
"""

DEFAULT_CAM_HEIGHT_M = 1.45   # phone held while walking, chest/eye level
MIN_CAM_HEIGHT_RATIO = 0.03   # camera height / scene extent below this = tabletop


def auto_scale_from_camera(cam_h_units, extent_units, assumed_m=DEFAULT_CAM_HEIGHT_M):
    """Return (scale_m_per_unit, note, confident).

    cam_h_units: median camera height above the floor, in model units.
    extent_units: scene diagonal in model units (for the plausibility guard).
    """
    if cam_h_units is None or cam_h_units <= 1e-6:
        return None, "ความสูงกล้อง ~0 — ประเมินสเกลอัตโนมัติไม่ได้", False

    scale = assumed_m / cam_h_units
    ratio = (cam_h_units / extent_units) if extent_units else 0.0
    confident = ratio >= MIN_CAM_HEIGHT_RATIO

    if confident:
        note = (f"สเกลประเมินอัตโนมัติจากความสูงกล้องสมมติ ~{assumed_m:.2f} m "
                f"— ตัวเลขเป็นเมตรโดยประมาณ (อาจคลาด ±20%) วัดจริงใน measure viewer เพื่อความแม่นยำ")
    else:
        note = ("ระนาบพื้นน่าจะเป็นผิวโต๊ะ (กล้องอยู่ต่ำผิดปกติ) — "
                "สเกลอัตโนมัติไม่น่าเชื่อถือ ควรวัดจริงใน measure viewer")
    return scale, note, confident
