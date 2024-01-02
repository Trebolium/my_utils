import subprocess
import sys
import os
import pdb
from my_utils import recursive_file_retrieval

# unfinished
def delete_smallest_files(trg_dir='./', byte_thresh=1000):
    bash_command = f'du -ah {trg_dir} | grep -v "\s/[^.]*$" | sort -rh'
    # 'du -ah ./ | grep -v "\s/[^.]*$" | sort -rh'
    process = subprocess.Popen(bash_command.split(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, _ = process.communicate()
    pdb.set_trace()
    h_string = output.decode()
    string_list_wo_tabs = h_string.replace('\t', ' ').split('\n')
    size_dir_list = []
    for entry in string_list_wo_tabs:
        split_entries = entry.split(' ')
        size = split_entries[0]
        if size.endswith('K'):
            pdb.set_trace()
            int(size[:-1])*1000
            size_dir_list.append()


def main():
    delete_smallest_files(sys.argv[1], int(sys.argv[2]))

if __name__ == '__main__':
    main()