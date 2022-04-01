import matplotlib.pyplot as plt
import numpy as np
import math

"""Determines what the correct axis is for viewing in matplotlib"""

def fix_axes(melspec, librosa_used):
    if librosa_used: melspec = np.flip(melspec, axis=-2) #librosa has all numpy flipped compared to intuitive reading
    else: melspec = np.rot90(melspec, axes=(-2,-1)) #if not used, timestamps are vertical and the axes must be rotated
    return melspec


""" Allows one to plot spectrograms

    Must be fed an np array of spectrograms (all same size)
    If it is fed a filepath string, this will be the location the plot is saved to
    list_of_strings provides it with x_labels, y_labels and all remaining titles
    num_cols determines how many columns the subplots will take form in.
"""
def plot_specs(array_of_specs, file_path=None, list_of_strings=None, num_cols=1, figsize=(20,5)):
    num_specs = array_of_specs.shape[0]
    num_feats = array_of_specs.shape[1]
    num_frames = array_of_specs.shape[2]
    x = np.arange(num_frames)
    num_rows = math.ceil(num_specs/num_cols)
    
    if list_of_strings == None: list_of_strings = ['No Title' for i in range(num_specs+2)]
    x_label = list_of_strings[0]
    y_label = list_of_strings[1]
    labels = list_of_strings[2:]
    
    plt.figure(figsize=figsize)
    for i in range(num_specs):
        this_array = array_of_specs[i]
        plt.subplot(num_rows, num_cols, i+1)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.imshow(this_array)
    plt.show()
    if not file_path == None:
        plt.savefig(file_path)



""" Plots contours such as pitch info or the performance of a model.
    Feed this an np array of one or more data vectors. 
    If it is fed a filepath string, this will be the location the plot is saved to
    list_of_strings provides it with x_labels, y_labels and all remaining titles
    num_cols determines how many columns the subplots will take form in.
"""
def plot_contours(array_of_contours, file_path=None, list_of_strings=None, num_cols=1, figsize=(20,5)):
    num_contours = array_of_contours.shape[0]
    num_steps = array_of_contours.shape[1]
    x = np.arange(num_steps)
    num_rows = math.ceil(num_contours/num_cols)
    
    if list_of_strings == None: list_of_strings = ['No Title' for i in range(num_contours+2)]
    x_label = list_of_strings[0]
    y_label = list_of_strings[1]
    labels = list_of_strings[2:]
    
    plt.figure(figsize=figsize)
    for i in range(num_contours):
        plt.subplot(num_rows, num_cols, i+1)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(x, array_of_contours[i], 'r--',label=labels[i])
        plt.legend()
    plt.show()
    if not file_path == None:
        plt.savefig(file_path)