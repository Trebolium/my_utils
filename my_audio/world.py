import numpy as np
import pyworld as pw
import pysptk, scipy, copy, warnings, pdb
from my_arrays import fix_feat_length
from my_audio.pitch import midi_as_onehot
# import soundfile as sf
# import librosa

# Alpha value used in 'Mel-generalized cepstral analysis
#   a unified approach to speech spectral estimation', referenced in Nercessian 2020.
#   Also used in chandna's code for WGANsing

def generate_params(use_chandna_config, sampling_rate, frame_dur_ms):
    if use_chandna_config:
        "Hard values set to variables as these are specific to the pyworld.wav2world() function which is used by Chandna"
        fft_size = 1024
        frame_dur_ms = 5.80498866 # chosen by Chandna, is the duration of frames in ms
        hopsize = sampling_rate / frame_dur_ms #hopsize is not explicitly defined by users in wav2world().
        feat_params = {'type':'chandna', 'fmin':None, "fmax":None, 'num_feats':44, 'frame_dur_ms':frame_dur_ms, 'sr':sampling_rate, 'fft_size':fft_size} # pw.default_frame_period is 5ms. We've changed the frame_dur to 10 ms
    else:
        hop_size = sampling_rate / frame_dur_ms
        fft_size = 1024 # Docs say Harvest determines fft_size as a function of sr and F0-min. However doesn't specifiy what this function is. determined size of 1024 in current application by looking at feature dims output size along freq axis
        feat_params = {'type':'harmsOnly', "fmin":50, "fmax":1100, 'num_feats':40, 'frame_dur_ms':frame_dur_ms, 'sr':sampling_rate, 'fft_size':fft_size} # pw.default_frame_dur_ms is 5ms. We've changed the frame_dur to 10 ms  
    return feat_params

def midi_to_worldf0(midi_voicings):
    pitched = midi_voicings[:,1]==0
    unpitched = ~pitched
    f0 = 2**((midi_voicings[:,0]-69)/12)*440
    f0[unpitched] = 0
    f0 = f0.astype('float64')
    return f0


def get_warped_freqs(ndim, sr, fw):
    # warped frequencies
    f = np.linspace(0.0, sr/2, ndim)
    w = 2*np.pi*f/sr
    a = -fw  # XXX: why negative?
    w_warped = np.arctan2((1 - a**2)*np.sin(w), (1 + a**2)*np.cos(w) - 2*a)
    f_warped = w_warped/(2*np.pi)*sr       
    return f_warped


def mfsc_to_sp(mfsc, spec_size, sr, alpha=0.45):
    # reconstruct sp by interpolation
    # alternative is mfsc_to_mgc(), mgc_to_sp() which uses SPTK's mgc2sp
    # interpolation is quite accurate if high mfsc dimensionality is used (e.g. 60)
    f_warped = get_warped_freqs(mfsc.shape[1], sr, fw=alpha)
    is_1d = mfsc.ndim == 1
    mfsc = np.atleast_2d(mfsc)
    ndim = mfsc.shape[1]

    f_sp = np.linspace(0, sr/2, spec_size)
    interp_f = scipy.interpolate.CubicSpline(f_warped, mfsc, axis=-1, bc_type='clamped', extrapolate=None)
    sp = interp_f(f_sp)

    if is_1d:
        sp = sp.flatten()
    
    return sp


"""sp_to_mgc, sp_to_mfsc and mgc_to_mfcs: All taken from https://github.com/Trebolium/WGANSing/blob/mtg/reduce.py"""
def sp_to_mgc(sp, ndim, fw, noise_floor_db=-120.0):
    # HTS uses -80, but we shift WORLD/STRAIGHT by -20 dB (so would be -100); use a little more headroom (SPTK uses doubles internally, so eps 1e-12 should still be OK)
    dtype = sp.dtype
    sp = sp.astype(np.float64)  # required for pysptk
    mgc = np.apply_along_axis(pysptk.mcep, 1, np.atleast_2d(sp), order=ndim-1, alpha=fw, maxiter=0, etype=1, eps=10**(noise_floor_db/10), min_det=0.0, itype=1)
    if sp.ndim == 1:
        mgc = mgc.flatten()
    mgc = mgc.astype(dtype)
    return mgc


def mgc_to_mfsc(mgc):
    is_1d = mgc.ndim == 1
    mgc = np.atleast_2d(mgc)
    ndim = mgc.shape[1]

    # mirror cepstrum
    mgc1 = np.concatenate([mgc[:, :], mgc[:, -2:0:-1]], axis=-1)

    # re-scale 'dc' and 'nyquist' cepstral bins (see mcep())
    mgc1[:, 0] *= 2
    mgc1[:, ndim-1] *= 2
    
    # fft, truncate, to decibels
    mfsc = np.real(np.fft.fft(mgc1))
    mfsc = mfsc[:, :ndim]
    mfsc = 10*mfsc/np.log(10)

    if is_1d:
        mfsc = mfsc.flatten()

    return mfsc


def sp_to_mfsc(sp, ndim, fw, noise_floor_db=-120.0):
    # helper function, sp->mgc->mfsc in a single step
    mgc = sp_to_mgc(sp, ndim, fw, noise_floor_db)
    mfsc = mgc_to_mfsc(mgc)
    return mfsc


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isinf(y), lambda z: z.nonzero()[0]

def freq_to_vuv_midi(f0, ignore_pitchless=True):
    #Convert to midi notes, with second vector displaying 1 when there's no pitch detected
    with warnings.catch_warnings(): # warning 
        warnings.simplefilter("ignore", category=RuntimeWarning)
        notes_y = 69+12*np.log2(f0/440)
    y = notes_y
    "Nan related"
    nans, x= nan_helper(y)
    if np.all(nans) == True:
        # if ignore_pitchless:
        #     raise ValueError('No voice pitch detected in segment')
        # else:
        y = np.stack((np.zeros((y.shape[0])), np.ones((y.shape[0]))), axis=-1)
        return y
    naners=np.isinf(y)
    # interpret pitch between voiced sections 
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    y=np.array(y).reshape([len(y),1])
    unvoiced=np.array(naners).reshape([len(y),1])
    y=np.concatenate((y,unvoiced),axis=-1)
    return y


def onehotmidi_from_world_fp(pitch_feat_path, offset, window_size, midi_range):
    pitch_pred = np.load(pitch_feat_path)[:,-2:]
    midi_contour = pitch_pred[:,0]
    # remove the interpretted values generated because of unvoiced sections
    unvoiced = pitch_pred[:,1].astype(int) == 1
    midi_contour[unvoiced] = 0

    if offset < 0:
        midi_trimmed, _ = fix_feat_length(midi_contour, window_size)
    else:
        midi_trimmed = midi_contour[offset:(offset+window_size)]
    onehot_midi = midi_as_onehot(midi_trimmed, midi_range)

    return onehot_midi


def mfsc_to_world_to_audio(harm_mfsc, ap_mfsc, midi_voicings, feat_params):
    # pdb.set_trace()
    # try:
    harm_sp = mfsc_to_sp(harm_mfsc, feat_params['fft_size']//2+1, feat_params['sr'])
    harm_sp_rescaled = 10**(harm_sp/10)
    ap_sp = mfsc_to_sp(ap_mfsc, feat_params['fft_size']//2+1, feat_params['sr'])
    ap_sp_rescaled = np.sqrt((10**(ap_sp/10))) 
    # if type(midi_voicings) == torch.Tensor:
    #     midi_voicings = midi_voicings.numpy()
    f0 = midi_to_worldf0(midi_voicings)
    harm_sp, ap_sp, f0 = np.ascontiguousarray(harm_sp_rescaled), np.ascontiguousarray(ap_sp_rescaled), np.ascontiguousarray(f0)
    synthed_audio = pw.synthesize(f0, harm_sp, ap_sp, feat_params['sr'], feat_params['frame_dur_ms'])
    # except Exception as e:
    #     pdb.set_trace()
    return synthed_audio


def code_harmonic(sp, order, alpha=0.45, mcepInput=3, en_floor=10 ** (-80 / 20)):
    """Taken from https://github.com/seaniezhao/torch_npss/blob/master/data/data_util.py"""
    #get mel cepstrum analysis
    # the first arugment (pysptk.mcep) in the apply_along_axis function is a function itself. The second is the axis upon which the input array is sliced. Third is input array. Rest are arguments for pysptk.mcep 
    mceps = np.apply_along_axis(pysptk.mcep, 1, sp, order - 1, alpha, itype=mcepInput, threshold=en_floor)
    #do fft and take real
    scale_mceps = copy.copy(mceps)
    scale_mceps[:, 0] *= 2
    scale_mceps[:, -1] *= 2
    mirror = np.hstack([scale_mceps[:, :-1], scale_mceps[:, -1:0:-1]])
    mfsc = np.fft.rfft(mirror).real
    return mfsc


def get_world_feats(y, feat_params):
    y = y.astype('double')
    if feat_params['w2w_process'] == 'wav2world':
        feats=pw.wav2world(y, feat_params['sr'],frame_period=feat_params['frame_dur_ms'])
        harm = feats[1]
        aper = feats[2]
        refined_f0 = feats[0]

    else:
        if feat_params['w2w_process'] == 'harvest':
            f0, t_stamp = pw.harvest(y, feat_params['sr'], feat_params['fmin'], feat_params['fmax'], feat_params['frame_dur_ms'])
        elif feat_params.w2w_process =='dio':
            f0, t_stamp = pw.dio(y, feat_params['sr'], feat_params['fmin'], feat_params['fmax'], frame_period = feat_params['frame_dur_ms'])
        refined_f0 = pw.stonemask(y, f0, t_stamp, feat_params['sr'])
        harm = pw.cheaptrick(y, refined_f0, t_stamp, feat_params['sr'], f0_floor=feat_params['fmin'])
        aper = pw.d4c(y, refined_f0, t_stamp, feat_params['sr'])

    refined_f0 = freq_to_vuv_midi(refined_f0) # <<< this can be done at training time
    
    if feat_params['dim_red_method'] == 'code-h':
        harm = code_harmonic(harm, feat_params['num_harm_feats'])
        aper = code_harmonic(aper, feat_params['num_aper_feats'])
    elif feat_params['dim_red_method'] == 'world':
        harm = pw.code_spectral_envelope(harm, feat_params['sr'], feat_params['num_harm_feats'])
        aper = pw.code_aperiodicity(aper, feat_params['sr'])
    elif feat_params['dim_red_method'] == 'chandna':
        harm = 10*np.log10(harm) # previously, using these logs was a separate optional process to 'chandna'
        aper = 10*np.log10(aper**2)
        harm = sp_to_mfsc(harm, feat_params['num_harm_feats'], 0.45)
        aper =sp_to_mfsc(aper, feat_params['num_aper_feats'], 0.45)
    else:
        raise Exception("The value for dim_red_method was not recognised")

    out_feats=np.concatenate((harm,aper,refined_f0),axis=1)

    return out_feats