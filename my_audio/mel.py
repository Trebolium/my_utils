import librosa, pdb, copy #, pysptk
import numpy as np
import soundfile as sf
import pyworld as pw
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel

prng = np.random.RandomState(1) 
mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))

""" Manually computign the STFT
    Code taken from repo at https://github.com/auspicious3000/autovc"""

""" Filter out unwanted frequencies by determining nyquist freq and defining a floor frequency (cutoff)"""
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

""" stft transform """
def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length//2), mode='reflect')
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    return np.abs(result)    

"""adds a highpass filter to audio, then injects small amount of noise"""
def add_butter_noise(audio, sr):
    prng = np.random.RandomState(1)
    b, a = butter_highpass(30, sr, order=5)
    y = signal.filtfilt(b, a, audio)
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    return wav

# mel_filter = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
"""applies stft transform to audio, then mel filter banks"""
def audio_to_mel_autovc(audio, fft_size, hop_length: int, mel_filter):
    D = pySTFT(audio, fft_size, hop_length).T
    db_unnormed_melspec = np.dot(D, mel_filter)
    return db_unnormed_melspec


def raw_audio_to_mel_autovc(y, mel_filter, min_level, hop_length, sr, fft_size):   
    y = add_butter_noise(y, sr)
    db_unnormed_melspec = audio_to_mel_autovc(y, fft_size, hop_length, mel_filter)
    autovc_mel = db_normalize(db_unnormed_melspec, min_level)
    return autovc_mel


# min_level = np.exp(-100 / 20 * np.log(10))
"""Converts amplitude to decibels and optionally normalises"""
def db_normalize(melspec, min_level, normalise=True):
    floored_mel = np.maximum(min_level, melspec) # creates a new array, clipping at the minimum_level
    db_melspec = 20 * np.log10(floored_mel) - 16 # converts to decibels (20*log10) and removes 16db for headroom
    if normalise:
        db_clipped_melspec = np.clip((db_melspec + 100) / 100, 0, 1) # Add 100 to ensure the minimal value is at least 0 before. Clip from 0 to 1 anyway
        return db_clipped_melspec
    else:
        return db_melspec

""" Achieves the same as audio_to_mel_autovc() but with librosa library"""
def audio_to_mel_librosa(wav, sr, n_fft_size, hop, n_mels=80, fmin=90, fmax=7600):
    melspec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft_size, hop_length=hop, n_mels=n_mels, fmin=fmin, fmax=fmax)
    db_melspec = librosa.amplitude_to_db(melspec,ref=np.max)
    default_axes_melspec = np.swapaxes(db_melspec, 0, 1)
    return default_axes_melspec
