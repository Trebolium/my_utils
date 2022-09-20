import os,random,pdb,shutil
from tqdm import tqdm

root = '/homes/bdoc3/my_data/world_vocoder_data/vctk_npss_world'
train_dir = root +'/train'
val_dir = root +'/val'
train_split = 0.8

_,dirs,_ = next(os.walk(root))

if not os.path.exists(train_dir): os.mkdir(train_dir)
if not os.path.exists(val_dir): os.mkdir(val_dir)

random.seed(1)

for dir in tqdm(dirs):
    pdb.set_trace()
    _,_,files = next(os.walk(dir))
    total = len(files)
    random.shuffle(files)    
    train_part = files[:int(total*train_split)]
    val_part = files[int(total*train_split):]
    if not os.path.exists(os.path.join(train_dir, dir)): os.mkdir(os.path.join(train_dir, dir))
    if not os.path.exists(os.path.join(val_dir, dir)): os.mkdir(os.path.join(val_dir, dir)) 
    for train_f in train_part:
        shutil.move(os.path.join(root, dir, train_f), os.path.join(train_dir, dir, train_f))
    for val_f in val_part:
        shutil.move(os.path.join(root, dir, val_f), os.path.join(val_dir, dir, val_f))
    