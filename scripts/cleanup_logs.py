'''
python scripts/cleanup_logs.py
'''

#!/usr/bin/env python3
import glob, os, shutil

ROOT = "./logs/"
ALLOWED = {".json", ".png", ".jpeg", '.jpg', }
DRY_RUN = False

dirs = sorted(glob.glob(os.path.join(ROOT, "**/"), recursive=True), key=lambda d: -d.count(os.sep))

print('Will delete (unless dry-running):')
for d in dirs:
    entries = glob.glob(os.path.join(d, "*"))
    if entries and all(os.path.isfile(e) and os.path.splitext(e)[1].lower() in ALLOWED or '.tfevents.' in e for e in entries):
        print(f"{glob.glob(f'{d}/**')}")
        if not DRY_RUN:
            assert all([any([f.endswith(a) or '.tfevents.' in f for a in ALLOWED]) for f in glob.glob(f'{d}/**')]), [f for f in glob.glob(f'{d}/**')]
            shutil.rmtree(d)


