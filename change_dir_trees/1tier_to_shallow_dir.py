import os, pdb, shutil

root_dir = './'
singer_list = []

_, dirs, _ = next(os.walk(root_dir))

for dir in dirs:
    _, _, files = next(os.walk(dir)) 
    for f in files:
        shutil.move(os.path.join(dir, f), f)
