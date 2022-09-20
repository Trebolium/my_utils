from ast import Assert
import numpy as np
import mpu.ml, pdb

"""convert midi contours to 1hot representation
not relevant to other 1hotting as 0 means unvoiced,
and treating outliers is specific for midi handling
"""
def midi_as_onehot(contour, cont_range):

    # make contour int numpy
    if type(contour) == list:
        contour = np.asarray(contour)

    # smoothen contour
    # contour = medfilt(contour,3)
    contour = contour.astype(int)
    contour_copy = np.copy(contour)

    # assert that input is within the range
    below_range = contour < min(cont_range) #bool
    # for elements that aren't 0 and less than midi min
    if np.all(contour[below_range] != 0): 
        not_zeros = np.where(contour !=0 )
        below_range = np.where(contour < min(cont_range))
        offending_indices = np.intersect1d(not_zeros, below_range)
        contour[offending_indices] = 0
    # for elements above midi max
    if np.max(contour) > max(cont_range):
        contour = np.clip(contour, 0, max(cont_range))

    # convert unvoiced to lowest value and map contour to acceptable range
    unvoiced = (contour==0)
    contour[unvoiced] = min(cont_range) - 1
    new_contour =  contour - (min(cont_range) - 1)

    # convert to 1hot encoding using length of range + unvoiced as num_classes
    onehot_enc_list = mpu.ml.indices2one_hot(new_contour, nb_classes = len(cont_range)+1)
    onehot_enc_arr = np.asarray(onehot_enc_list)

    return onehot_enc_arr


if __name__ == '__main__':
    contour = [69.34, 44.12, 33, 0, 0, 80, 55]
    cont_range = range(10,88)
    test = midi_as_onehot(contour, cont_range)

