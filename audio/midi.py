# from https://gist.github.com/YuxiUx/ef84328d95b10d0fcbf537de77b936cd
from math import log2


def noteToFreq(note):
    midi_note = 12 * log2(note/440) + 69
    return midi_note