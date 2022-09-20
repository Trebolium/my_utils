import os, pickle


corruptions_found = True

while corruptions_found:
    # os.system("python dataset_64_to_32_np.py /homes/bdoc3/my_data/world_vocoder_data/vctk 16")
    # os.system("count_corrupt.py /homes/bdoc3/my_data/world_vocoder_data/m4a2worldChandna")
    corrupt_list = pickle.load(open('corrupt_numpy.pkl','rb'))
    if len(corrupt_list) == 0:
       corruptions_found  = False
    else:
        os.system("python regenerate_corrupt.py -p=corrupt_numpy.pkl -fs=1024")

