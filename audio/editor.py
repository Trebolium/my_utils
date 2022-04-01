import numpy as np
import scipy, librosa, pdb

# analyses audio loudness to determine of an instrument is playing
def compute_activation_confidence(
    audio, rate, win_len=4096, lpf_cutoff=0.075, theta=0.15, var_lambda=20.0, amplitude_threshold=0.01
):
    """Create the activation confidence annotation for a multitrack. The final
    activation matrix is computed as:
        `C[i, t] = 1 - (1 / (1 + e**(var_lambda * (H[i, t] - theta))))`
    where H[i, t] is the energy of stem `i` at time `t`

    Parameters
    ----------
    track : Audio path
    win_len : int, default=4096
        Number of samples in each window
    lpf_cutoff : float, default=0.075
        Lowpass frequency cutoff fraction
    theta : float
        Controls the threshold of activation.
    var_labmda : float
        Controls the slope of the threshold function.
    amplitude_threshold : float
        Energies below this value are set to 0.0

    Returns
    -------
    C : np.array
        Array of activation confidence values shape (n_conf, n_stems)
    stem_index_list : list
        List of stem indices in the order they appear in C

    """
    H = []
    # MATLAB equivalent to @hanning(win_len)
    win = scipy.signal.windows.hann(win_len + 2)[1:-1]

    # audio, rate = librosa.load(track, mono=True)
    H.append(track_energy(audio.T, win_len, win))

    # list to numpy array
    H = np.array(H)

    # normalization (to overall energy and # of sources)
    E0 = np.sum(H, axis=0)

    H = H / np.max(E0)
    # binary thresholding for low overall energy events
    H[:, E0 < amplitude_threshold] = 0.0

    # LP filter
    b, a = scipy.signal.butter(2, lpf_cutoff, "low")
    H = scipy.signal.filtfilt(b, a, H, axis=1)

    # logistic function to semi-binarize the output; confidence value
    C = 1.0 - (1.0 / (1.0 + np.exp(np.dot(var_lambda, (H - theta)))))

    # add time column
    time = librosa.core.frames_to_time(np.arange(C.shape[1]), sr=rate, hop_length=win_len // 2)

    # stack time column to matrix
    C_out = np.vstack((time, C))
    # print(C_out.T)
    return C_out.T


def track_energy(wave, win_len, win):
    """Compute the energy of an audio signal

    Parameters
    ----------
    wave : np.array
        The signal from which to compute energy
    win_len: int
        The number of samples to use in energy computation
    win : np.array
        The windowing function to use in energy computation

    Returns
    -------
    energy : np.array
        Array of track energy

    """
    hop_len = win_len // 2
    wave = np.lib.pad(wave, pad_width=(win_len - hop_len, 0), mode="constant", constant_values=0)

    # post padding
    wave = librosa.util.fix_length(wave, int(win_len * np.ceil(len(wave) / win_len)))

    # cut into frames
    wavmat = librosa.util.frame(wave, frame_length=win_len, hop_length=hop_len)

    # Envelope follower
    wavmat = hwr(wavmat) ** 0.5  # half-wave rectification + compression
    return np.mean((wavmat.T * win), axis=1)

# generates list of chunks from audio using activity likelihood (conf) and duration threshold
def audiochunks_from_conf(audio, rate, conf, act_thresh=0.9, time_thresh=3.2):
    time_durs = []
    audio_chunks = []
    act_on = False
    for i, entry in enumerate(conf):
        if entry[1]>=act_thresh:
            if act_on == False:
                last_act_on_time = entry[0]
                act_on = True
        else:
            if act_on == True:
                duration = entry[0] - last_act_on_time
                if duration >= time_thresh:
                    time_durs.append((last_act_on_time,duration))
                    start_samp = int(last_act_on_time*rate)
                    end_samp = int(entry[0]*rate)
                    audio_chunk = audio[start_samp:end_samp]
                    audio_chunks.append(audio_chunk)
                act_on = False
    return audio_chunks

def hwr(x):
    """ Half-wave rectification.

    Parameters
    ----------
    x : array-like
        Array to half-wave rectify

    Returns
    -------
    x_hwr : array-like
        Half-wave rectified array

    """
    return (x + np.abs(x)) / 2

def path_to_damp_format(file_path, feat_params):
    audio, rate = librosa.load(file_path, sr=feat_params['sr'])
    conf = compute_activation_confidence(audio, rate)
    audio_chunks = audiochunks_from_conf(audio, rate, conf)
    cat_audio = np.asarray([samp for audio_chunk in audio_chunks for samp in audio_chunk]) #crude, but as there are so few splits anticipated   
    return cat_audio

def desilence_concat_audio(audio, rate):
    conf = compute_activation_confidence(audio, rate)
    audio_chunks = audiochunks_from_conf(audio, rate, conf)
    cat_audio = np.asarray([samp for audio_chunk in audio_chunks for samp in audio_chunk]) #crude, but as there are so few splits anticipated   
    return cat_audio

import sys, os, pdb
import soundfile as sf

if __name__ == '__main__':
    file_path = sys.argv[1]
    dst_f_path = os.path.join(sys.argv[2], os.path.basename(file_path)[:-4] +'_concat.wav') 
    sr = int(sys.argv[3])
    pdb.set_trace()
    y, _ = librosa.load(file_path, sr=sr)

    y = desilence_concat_audio(y, sr)
    
    sf.write(dst_f_path, y, samplerate=sr)