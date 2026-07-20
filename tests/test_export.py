"""Local export: copy a finished scan's deliverables to a permanent folder.

Run:  .venv-amd/Scripts/python.exe -X utf8 tests/test_export.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from lingbot_plus.export import export_scan, DELIVERABLES

    with tempfile.TemporaryDirectory() as td:
        scan = os.path.join(td, "myscan")
        os.makedirs(os.path.join(scan, "frames"))
        # deliverables + noise that must NOT be copied
        for f in ("report.html", "mesh.glb", "plan.png"):
            open(os.path.join(scan, f), "w").close()
        open(os.path.join(scan, "merged.npz"), "w").close()          # heavy, skip
        open(os.path.join(scan, "chunk_000.npz"), "w").close()       # heavy, skip
        open(os.path.join(scan, "frames", "000001.jpg"), "w").close()

        dest = os.path.join(td, "out")
        out, copied = export_scan(scan, dest_root=dest)

        assert os.path.basename(out) == "myscan", f"wrong folder name: {out}"
        assert set(copied) == {"report.html", "mesh.glb", "plan.png"}, f"copied {copied}"
        for f in copied:
            assert os.path.exists(os.path.join(out, f)), f"missing {f}"
        # never copies heavy intermediates or frames
        assert not os.path.exists(os.path.join(out, "merged.npz")), "copied merged.npz"
        assert not os.path.exists(os.path.join(out, "frames")), "copied frames dir"
        # readme present
        assert any(x.endswith(".txt") for x in os.listdir(out)), "no readme"
        # every DELIVERABLES entry is a bare filename (no path traversal)
        assert all("/" not in f and "\\" not in f for f in DELIVERABLES)

    print(f"OK — exported {sorted(copied)} to {os.path.basename(out)}/")


if __name__ == "__main__":
    main()
