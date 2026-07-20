"""Frame listing shared by chunked.py and worker.py.

Windows filesystems are case-insensitive, so globbing an extension list like
".jpg,.png,.JPG" matches every .jpg file twice (once via '*.jpg', once via
'*.JPG') — the reconstruction then ingests each frame twice back-to-back, which
destroys parallax between the duplicates. list_images dedupes case-insensitively
and returns a stable, name-sorted list.
"""

import glob
import os


def list_images(image_folder, image_ext=".jpg,.png,.JPG"):
    seen = set()
    paths = []
    for ext in image_ext.split(","):
        for p in glob.glob(os.path.join(image_folder, f"*{ext}")):
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen:
                seen.add(key)
                paths.append(p)
    return sorted(paths, key=lambda p: os.path.basename(p).lower())
