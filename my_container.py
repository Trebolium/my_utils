import os, sys, pdb, copy
import numpy as np

"""List manipulations"""

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

# separate list into multiple lists by string
def separate_by_strings(full_list, string_list):
    list_of_lists = [[] for i in range(len(string_list))]
    for s_idx, group in enumerate(string_list):
        this_string_list = []
        for file in full_list:
            if os.path.basename(file).startswith(group):
                this_string_list.append(file)
        list_of_lists[s_idx].extend(this_string_list)
    return list_of_lists

"""Misc"""

def str2bool(v):
    return v.lower() in ('true')