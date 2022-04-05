from scipy import medfilt
import numpy as np
import mpu.ml

def midi_as_onehot(midi_contour, midi_range):
    # get bool masks to treat unvoiced entirely separate to rest of frequency predictions
    midi_contour = np.round(midi_contour)
    midi_contour = medfilt(midi_contour,3)
    unvoiced = (midi_contour==0.)
    # convert those with low confidence to nans and use bool maks for processing
    midi_contour[unvoiced] = np.nan
    num_steps = len(midi_contour)
    onehot_enc_arr = mpu.ml.indices2one_hot(midi_contour, nb_classes = len(midi_range))
    # onehot_enc_arr = np.zeros((num_steps, len(midi_range)))
    # for i, midi_note in enumerate(midi_contour):
    #     onehot_enc_arr[i,midi_note] = 1
    return onehot_enc_arr

    # https://github.com/auspicious3000/autovc/issues/50 has some advice on this process
    vector_257_normalized_unit_var_voiced_log_freq[voiced_bool] = np.rint(midi_contour[~unvoiced]*midi_range)+1
    vector_257_vuv_normalized_unit_var_log_freq = vector_257_normalized_unit_var_voiced_log_freq.copy()
    vector_257_vuv_normalized_unit_var_log_freq[unvoiced_bool] = vector_257_vuv_normalized_unit_var_log_freq[unvoiced_bool]=0
    vector_257_vuv_normalized_unit_var_log_freq = vector_257_vuv_normalized_unit_var_log_freq.astype(int)
    one_hot_preprocessed_pitch_conotours = np.zeros((vector_257_vuv_normalized_unit_var_log_freq.size, vector_257_vuv_normalized_unit_var_log_freq.max()+1))
    one_hot_preprocessed_pitch_conotours[np.arange(vector_257_vuv_normalized_unit_var_log_freq.size),vector_257_vuv_normalized_unit_var_log_freq] = 1
    return one_hot_preprocessed_pitch_conotours 