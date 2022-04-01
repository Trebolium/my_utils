import numpy as np
from scipy import signal

""" Manually computign the STFT
    Code taken from repo at https://github.com/auspicious3000/autovc"""

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length//2), mode='reflect')
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    fft_window = signal.get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    return np.abs(result)  

"""adds a highpass filter to audio, then injects small amount of noise"""
def add_butter_noise(audio, sr):
    prng = np.random.RandomState(1)
    b, a = butter_highpass(30, sr, order=5)
    y = signal.filtfilt(b, a, audio)
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    return wav

# mel_filter = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
"""applies stft transform to audio, then mel filter banks"""
def audio_to_mel_autovc(audio, fft_size, hop_size, mel_filter):
    D = pySTFT(audio, fft_size, hop_size).T
    melspec = np.dot(D, mel_filter)
    return melspec

# min_level = np.exp(-100 / 20 * np.log(10))
"""Converts amplitude to decibels and normalises"""
def db_normalize(melspec, min_level, normalise=True):
    floored_mel = np.maximum(min_level, melspec) # creates a new array, clipping at the minimum_level
    db_melspec = 20 * np.log10(floored_mel) - 16 # converts to decibels (20*log10) and removes 16db for headroom
    if normalise:
        db_clipped_melspec = np.clip((db_melspec + 100) / 100, 0, 1) # Add 100 to ensure the minimal value is at least 0 before. Clip from 0 to 1 anyway
        return db_clipped_melspec
    else:
        return db_melspec