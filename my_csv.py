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


