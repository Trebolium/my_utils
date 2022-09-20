import pickle, os, pdb
import numpy as np

qwe = pickle.load(open('corrupt_numpy.pkl', 'rb'))

pdb.set_trace()
for tup in qwe:
    cp = tup[1]
    os.remove(cp)
    cfd = os.path.dirname(cp)
    if len(os.listdir(cfd)) == 0:
        os.rmdir(cfd)