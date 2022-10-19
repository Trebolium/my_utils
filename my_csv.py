import csv


# saves a python list as a csv file
def list_to_csvfile(list_of_rows, fp):
    with open(fp, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(list_of_rows)
            
            
# converts a csv file to a python list of rows
def csvfile_to_list(fp):
    f = open(fp)
    reader = csv.reader(f)
    _ = next(reader)
    list_of_rows = [row for row in reader]
    return list_of_rows


def vctk_id_gender_list(csv_path='/homes/bdoc3/my_data/text_data/vctk/speaker-info.txt'):   
    f = open(csv_path, 'r')
    header = f.readline()
    lines = f.readlines()
    id_list = []
    gender_list = []
    for line in lines:
        line_elements = [el for el in line.split(' ') if el!='']
        id_list.append(line_elements[0])
        gender_list.append(line_elements[2])
    return id_list, gender_list