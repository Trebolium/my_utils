import os, pdb, shutil, sys

root_dir = sys.argv[1]

"Get subset split dirs"
_, dirs, _ = next(os.walk(root_dir))
pdb.set_trace()
for subset in dirs:
    _, voices_dirs, _ = next(os.walk(os.path.join(root_dir, subset)))
    for voice_dir in voices_dirs:
        _, _, files = next(os.walk(os.path.join(root_dir, subset, voice_dir)))
        for f in files:
            shutil.move(os.path.join(root_dir, subset, voice_dir, f), os.path.join(root_dir, subset, f))
        os.rmdir(os.path.join(root_dir, subset, voice_dir))