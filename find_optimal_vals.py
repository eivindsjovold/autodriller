import numpy as np
from models.eckels.eckel import rate_of_penetration_eckel_vary_founder_wob, rate_of_penetration_eckel_vary_founder_rpm, rate_of_penetration_eckel_vary_founder_flow, rate_of_penetration_eckel_vary_founder
from models.bourgouyne_young_1974.rate_of_penetration_v3 import f5, f6, f8, rate_of_penetration_modv3
import csv
import random
case = 'BY'


if case == 'eckel2':
    print(case)
    wob_dict = []
    rop_dict = []
    rpm_dict =[]
    q_dict = []
    K = [0.26767342123727805, 0.4324315229681549, 0.6132226175113474, 0.21337671158654828, 0.3319251595736624, 0.31953446975649136, 0.7508770701029669, 0.23063047208802645, 0.47994829875469297, 0.5043919216105951, 0.5351330724339952, 0.32804166681464336, 0.4832090904711962, 0.8732485278109159, 0.5972075094846072, 0.5966112088465494, 0.7782560909631269, 0.6975594101438636, 0.5999282569783442, 0.502991005323338, 0.025, 0.9, 0.025]
    #K = [0.025, 0.9]
    a = 1
    b = 0.75
    c = 0.5
    k = 0.1
    rho = 1000
    d_n = 0.25
    my = 0.4
    a11 = 0.005
    a22 = 0.005
    a33 = 0.005

    with open('testing_vals.txt', 'w') as textfile:
        for k in range(len(K)):
            drillabililty = K[k]
            wob_threshold = -9999
            rpm_threshold = -9999
            flow_threshold = -9999
            wob_index = 0
            rpm_index = 0
            flow_index = 0
            rop_optimal = 0

            for i in range(1,400):
                wob = i
                rpm = i
                q = i
                rop_wob = rate_of_penetration_eckel_vary_founder_wob(a,b,c,drillabililty, k, wob, rpm, q, rho, d_n, my, a11, a22, a33)
                rop_rpm = rate_of_penetration_eckel_vary_founder_rpm(a,b,c,drillabililty, k, wob, rpm, q, rho, d_n, my, a11, a22, a33)
                rop_q = rate_of_penetration_eckel_vary_founder_flow(a,b,c,drillabililty, k, wob, rpm, q, rho, d_n, my, a11, a22, a33)
                if rop_wob > wob_threshold:
                    wob_threshold = rop_wob
                    wob_index = i

                if rop_rpm > rpm_threshold:
                    rpm_threshold = rop_rpm
                    rpm_index = i
                if rop_q > flow_threshold:
                    flow_threshold = rop_q
                    flow_index = i
            rop_optimal = rate_of_penetration_eckel_vary_founder(a,b,c,drillabililty, k, wob_index, rpm_index, flow_index, rho, d_n, my, a11, a22, a33)
            textfile.write(' K: ')
            textfile.write(str(drillabililty))
            textfile.write(' WOB: ')
            textfile.write(str(wob_index))
            textfile.write(' RPM: ')
            textfile.write(str(rpm_index))
            textfile.write(' q: ')
            textfile.write(str(flow_index))
            textfile.write(' ROP: ')
            textfile.write(str(rop_optimal))
            textfile.write('\n')
    textfile.close()


    with open('optimal_values.csv', 'w', newline = '') as csvfile:
        for k in range(len(K)):
            drillabililty = K[k]
            wob_threshold = -9999
            rpm_threshold = -9999
            flow_threshold = -9999
            wob_index = 0
            rpm_index = 0
            flow_index = 0
            rop_optimal = 0

            for i in range(1,400):
                wob = i
                rpm = i
                q = i
                rop_wob = rate_of_penetration_eckel_vary_founder_wob(a,b,c,drillabililty, k, wob, rpm, q, rho, d_n, my, a11, a22, a33)
                rop_rpm = rate_of_penetration_eckel_vary_founder_rpm(a,b,c,drillabililty, k, wob, rpm, q, rho, d_n, my, a11, a22, a33)
                rop_q = rate_of_penetration_eckel_vary_founder_flow(a,b,c,drillabililty, k, wob, rpm, q, rho, d_n, my, a11, a22, a33)
                if rop_wob > wob_threshold:
                    wob_threshold = rop_wob
                    wob_index = i

                if rop_rpm > rpm_threshold:
                    rpm_threshold = rop_rpm
                    rpm_index = i
                if rop_q > flow_threshold:
                    flow_threshold = rop_q
                    flow_index = i
            rop_optimal = rate_of_penetration_eckel_vary_founder(a,b,c,drillabililty, k, wob_index, rpm_index, flow_index, rho, d_n, my, a11, a22, a33)
            valuewriter = csv.writer(csvfile, delimiter =',')
            valuewriter.writerow([drillabililty,wob_index,rpm_index,flow_index,rop_optimal])
    csvfile.close()


elif case == 'BY':
    print(case)
    wob_dict = []
    rop_dict = []
    rpm_dict =[]
    q_dict = []
    rho = 22 
    wob_init = 0.1 #10^3 lbf/in 
    gp = 1.0        #lbm/gal
    db = 1.0
    db_init = db    #bit outer diameter in inches
    delta_t = 1/3600 #hrs
    h = 0.0001
    v = 10  #feet/s
    a11 = 0.005
    a22 = 0.005
    a33 = 0.005
    depth = 0

    wob = 10
    rpm = 10
    q = 10
    a1 = []
    a2 = 0
    a3 = 0
    a4 = 0
    a5 = []
    a6 = []
    a7 = 0
    a8 = []
    length = []

    with open('parameters_for_BY.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = ',')
        for row in csv_reader:
            a1.append(float(row[0]))
            #a5.append(float(row[1]))
            #a6.append(float(row[2]))
            #a8.append(float(row[3]))
            a5.append(2-float(row[0]))
            a6.append(float(row[0])*0.5)
            a8.append(float(row[0])-0.9)
            length.append(int(row[4]))              



    with open('optimal_value_BY.csv', 'w', newline = '') as csvfile:
        print('opened csvfile')
        csvwriter = csv.writer(csvfile, delimiter = ',')
        for x in range(len(a1)):
            wob_threshold = -9999
            rpm_threshold = -9999
            flow_threshold = -9999
            wob_index = 0
            rpm_index = 0
            flow_index = 0
            rop_optimal = 0
            for i in range(1,400):
                wob = i
                rpm = i
                q = i
                rop_wob = f5(a5[x], wob, wob_init, db, db_init,a11)
                rop_rpm = f6(a6[x], rpm, a22)
                rop_q = f8(a8[x], rho, q, v, a33)
                if rop_wob > wob_threshold:
                    wob_threshold = rop_wob
                    wob_index = i
                if rop_rpm > rpm_threshold:
                    rpm_threshold = rop_rpm
                    rpm_index = i
                if rop_q > flow_threshold:                    
                    flow_threshold = rop_q
                    flow_index = i
            rop_optimal = rate_of_penetration_modv3(a1[x],a2,a3,a4,a5[x],a6[x],a7,a8[x],depth,gp,rho,wob_index,wob_init,db,db_init,rpm_index,h,flow_index,v, a11, a22, a33)
            csvwriter.writerow([a1[x],a5[x],a6[x],a8[x],wob_index, rpm_index, flow_index, rop_optimal])
    csvfile.close()

elif case == 'generate_params':
    print(case)
    set_len = 10
    with open('parameters_for_BY.csv', 'w', newline = '') as csvfile:
        print('opened csvfile')
        csvwriter = csv.writer(csvfile, delimiter = ',')
        for i in range(set_len):
            a1 = random.uniform(1,1.5)
            a5 = random.uniform(0.5,1)
            a6 = random.uniform(0.4,0.1)
            a8 = random.uniform(0.3,0.6)
            length = random.randint(50,250)
            csvwriter.writerow([a1,a5,a6,a8,length])
    csvfile.close()