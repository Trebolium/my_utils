# from https://gist.github.com/YuxiUx/ef84328d95b10d0fcbf537de77b936cd
from math import log2
import numpy as np
ref_freq = 440
ref_midi = 69

def freqToMidi(freq):
    if type(freq) == np.ndarray:
        midi_note = 12 * np.log2(freq/ref_freq) + ref_midi
    else: 
        midi_note = 12 * log2(freq/ref_freq) + ref_midi
    return midi_note

def midiToFreq(midi_note):
    freq = 2**((midi_note-ref_midi)/12) * ref_freq
    return freq
