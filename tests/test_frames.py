"""Windows case-insensitive glob double-counts frames (.jpg matched by both
'*.jpg' and '*.JPG'). list_images must return each file once, sorted.

Run:  .venv-amd/Scripts/python.exe -X utf8 tests/test_frames.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from lingbot_plus.frames import list_images

    with tempfile.TemporaryDirectory() as td:
        for i in range(1, 6):
            open(os.path.join(td, f"{i:06d}.jpg"), "w").close()
        open(os.path.join(td, "cover.png"), "w").close()

        paths = list_images(td, ".jpg,.png,.JPG")
        # 5 jpg + 1 png = 6 unique, NOT 11 (jpg double-counted by .jpg and .JPG)
        assert len(paths) == 6, f"double-counted: {len(paths)} (want 6)"
        assert len(paths) == len(set(os.path.normcase(p) for p in paths)), "dupes present"
        # sorted, stable order
        assert paths == sorted(paths, key=lambda p: os.path.basename(p).lower()), "not sorted"
    print(f"OK — list_images deduped to 6 unique frames")


if __name__ == "__main__":
    main()
