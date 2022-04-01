import numpy as np
import pyworld as pw
import pysptk, scipy, copy, warnings, pdb


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

def freq_to_vuv_midi(f0):
    #Convert to midi notes, with second vector displaying 1 when there's no pitch detected
    with warnings.catch_warnings(): # warning 
        warnings.simplefilter("ignore", category=RuntimeWarning)
        notes_y = 69+12*np.log2(f0/440)
    y = notes_y
    "Nan related"
    nans, x= nan_helper(y)
    if np.all(nans) == True:
        raise ValueError('No voice pitch detected in segment')
    naners=np.isinf(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    # y=[float(x-(min_note-1))/float(max_note-(min_note-1)) for x in y]
    y=np.array(y).reshape([len(y),1])
    guy=np.array(naners).reshape([len(y),1])
    y=np.concatenate((y,guy),axis=-1)
    return y

def chandna_feats(audio, feat_params, include_f0=True, mode='mfsc'):
    feats=pw.wav2world(audio, feat_params['sr'],frame_period=feat_params['frame_dur_ms'])
    ap = 10*np.log10(feats[2]**2)
    harm=10*np.log10(feats[1])
    f0 = feats[0]
    y = freq_to_vuv_midi(f0)

    if mode == 'mfsc':
        harmy=sp_to_mfsc(harm,40,0.45)
        apy=sp_to_mfsc(ap,4,0.45)
    elif mode == 'mgc':
        harmy=sp_to_mgc(harm,60,0.45)
        apy=sp_to_mgc(ap,4,0.45)

    out_feats=np.concatenate((harmy,apy,y),axis=1)

    return out_feats


def mfsc_to_world_to_audio(harm_mfsc, ap_mfsc, midi_voicings, feat_params):
    harm_sp = mfsc_to_sp(harm_mfsc, feat_params['fft_size']//2+1, feat_params['sr'])
    harm_sp_rescaled = 10**(harm_sp/10)
    ap_sp = mfsc_to_sp(ap_mfsc, feat_params['fft_size']//2+1, feat_params['sr'])
    ap_sp_rescaled = np.sqrt((10**(ap_sp/10))) 
    f0 = midi_to_worldf0(midi_voicings)
    harm_sp, ap_sp, f0 = np.ascontiguousarray(harm_sp_rescaled), np.ascontiguousarray(ap_sp_rescaled), np.ascontiguousarray(f0)
    synthed_audio = pw.synthesize(f0, harm_sp, ap_sp, feat_params['sr'], feat_params['frame_dur_ms'])
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


def get_seaniezhao_feats(file_path, feat_params, use_npss):
    """Process used from https://github.com/seaniezhao/torch_npss/blob/master/data/preprocess.py"""
    x, org_sr = sf.read(file_path)
    if org_sr != feat_params['sr']:
        y = librosa.resample(x, org_sr, feat_params['sr'])
    f0, t_stamp = pw.harvest(y, feat_params['sr'], feat_params['fmin'], feat_params['fmax'], feat_params['frame_dur_ms'])
    refined_f0 = pw.stonemask(y, f0, t_stamp, feat_params['sr'])
    spec_env = pw.cheaptrick(y, refined_f0, t_stamp, feat_params['sr'], f0_floor=feat_params['fmin'])
    if use_npss == True:
        spec_env = code_harmonic(spec_env, feat_params['num_feats'])
    else:
        spec_env = pw.code_spectral_envelope(spec_env, feat_params['sr'], feat_params['num_feats'])
    aper_env = pw.d4c(y, refined_f0, t_stamp, feat_params['sr'])
    ap_env_reduced = pw.code_aperiodicity(aper_env, feat_params['sr'])
    return refined_f0, spec_env, aper_env, ap_env_reduced


def gen_world_feat(y, feat_params):
    """Process used from https://github.com/seaniezhao/torch_npss/blob/master/data/preprocess.py"""
    f0, t_stamp = pw.harvest(y, feat_params['sr'], feat_params['fmin'], feat_params['fmax'], feat_params['frame_dur_ms'])
    refined_f0 = pw.stonemask(y, f0, t_stamp, feat_params['sr'])
    spec_env = pw.cheaptrick(y, refined_f0, t_stamp, feat_params['sr'], f0_floor=feat_params['fmin'])
    spec_env = code_harmonic(spec_env, feat_params['num_feats'])
    # aper_env = pw.d4c(y, refined_f0, t_stamp, feat_params['sr'])
    # ap_env_reduced = pw.code_aperiodicity(aper_env, feat_params['sr'])
    return spec_env