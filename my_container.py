import os, sys, pdb, copy, csv
import ast
import numpy as np

"""List manipulations"""

def string_of_list_to_list(stringlist):
    """Converts a string representation of a list to a python list object"""
    return ast.literal_eval(stringlist)

# takes a list of substrings and removes any entry the main list that contains these substrings
def substring_exclusion(main_list, exclude_list):
    filtered_list = [] 
    for f_path in main_list:
        exclusion_found = False
        for exclusion in exclude_list:
            if exclusion in f_path:
                exclusion_found = True
        if exclusion_found == False: 
            filtered_list.append(f_path)
    return filtered_list

# tales a list and returns only those that are in the include_list
def substring_inclusion(main_list, include_list):
    if type(include_list) == str:
        include_list = [include_list]
    filtered_list = [] 
    for f_path in main_list:
        inclusion_found = False
        for inclusion in include_list:
            if inclusion in f_path:
                inclusion_found = True
        if inclusion_found == True: 
            filtered_list.append(f_path)
    return filtered_list

# ensures the total number of elements containing strings from the string_list are no more than the max allowance
def balance_by_strings(main_list, string_list, max_occurances):
    class_counter_list = np.zeros(len(string_list))
    balanced_file_list = []
    for i, file in enumerate(main_list):
        for class_idx, class_name in enumerate(string_list):
            if class_name in file:
                if class_counter_list[class_idx] >= max_occurances:
                    break 
                class_counter_list[class_idx] += 1
                balanced_file_list.append(file)
    return balanced_file_list

# separate list into multiple lists based on starting substrings
def separate_by_starting_substring(full_list, group_list):
    list_of_lists = [[] for i in range(len(group_list))]
    for s_idx, group in enumerate(group_list):
        this_group_list = []
        for file in full_list:
            if os.path.basename(file).startswith(group):
                this_group_list.append(file)
        list_of_lists[s_idx].extend(this_group_list)
    return list_of_lists


def reorder_by_classes(unordered_container, unordered_labels):
    """
    Reorder unordered_labels based on its numerical contents
    Reorder unordered_container based on unordered_label's numerical contents
    
    Return:
        Reordered lists
    """
    
    # ordered_indices = sorted(range(len(unordered_labels)), key=lambda k: unordered_labels[k])
    ordered_indices = np.argsort(unordered_labels)
    ordered_container = np.asarray([unordered_container[i] for i in ordered_indices])
    ordered_labels = np.asarray([unordered_labels[idx] for idx in ordered_indices])   
    
    return ordered_container, ordered_labels


def reorder_truncate(data_arr, labels, max_num_labels):
    """
    Does:
        Reorder data_arr and labels as described in reorder_by_classes()
        Truncate these based on max_num_labels threshold

    Return:
        Reordered data_arr and labels, truncated by threshold
    """
    ordered_embs, ordered_labels = reorder_by_classes(data_arr, labels)
    ordered_embs = ordered_embs[np.where(ordered_labels < max_num_labels)]
    ordered_labels = ordered_labels[np.where(ordered_labels < max_num_labels)]
    ordered_labels = np.expand_dims(ordered_labels, axis=1)
    return ordered_embs, ordered_labels


def flatten_and_label(list_of_lists):
    """
    Converts a grouped list to an array
    Provides an array of labels reflecting the index of these entries
    """
    
    grouped_list_arr = np.asarray([entry for this_list in list_of_lists for entry in this_list])
    
    group_labels = []
    for i, this_list in enumerate(list_of_lists):
        group_labels.extend([i] * len(this_list)) 
    group_labels_arr = np.asarray(group_labels)
    
    return grouped_list_arr, group_labels_arr
