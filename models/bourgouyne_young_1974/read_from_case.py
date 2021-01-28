import csv

def read_from_case():
    with open('bourgouyne_young_1974\\case.csv', newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        header = next(reader)
        depth_change =[]
        a1_vals = []
        a2_vals = []
        a3_vals = []
        a4_vals = []
        a5_vals = []
        a6_vals = []
        a7_vals = []
        a8_vals = []
        
        for row in reader:
            depth_change.append(float(row[0]))
            a1_vals.append(float(row[1]))
            a2_vals.append(float(row[2]))
            a3_vals.append(float(row[3]))
            a4_vals.append(float(row[4]))
            a5_vals.append(float(row[5]))
            a6_vals.append(float(row[6]))
            a7_vals.append(float(row[7]))
            a8_vals.append(float(row[8]))

    csvfile.close()
    file_information = []
    file_information.append(depth_change)
    file_information.append(a1_vals)
    file_information.append(a2_vals)
    file_information.append(a3_vals)
    file_information.append(a4_vals)
    file_information.append(a5_vals)
    file_information.append(a6_vals)
    file_information.append(a7_vals)
    file_information.append(a8_vals)

    return file_information
