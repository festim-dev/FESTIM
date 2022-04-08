import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv

types = ["A", "B"]
betas = [2, 5, 10, 15]

for charging_type in types:
    plt.figure()
    for beta in betas:

        T_exp = []
        flux_left = []
        flux_right = []
        with open('Solution/type_'+charging_type+'/beta='+str(beta)+'Kmin-1/derived_quantities.csv', 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                if 't(s)' not in row:
                    if float(row[3]) > 300:
                        T_exp.append(float(row[3]))
                        flux_left.append(float(row[1])*(1+1*(charging_type=="A")))
                        flux_right.append(float(row[2]))

        plt.plot(T_exp, -np.asarray(flux_left)-np.asarray(flux_right),
                 label=str(beta) + 'K min-1')

    plt.xlim(300, 600)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.3)
    plt.grid(which='major', alpha=0.7)
    plt.xlabel('T(K)')
    plt.ylabel(r'desorption(m$ ^{2}$/s)')
    plt.title('TDS type ' + charging_type)
    plt.legend()

# for beta in [2, 10, 15]:
T_exp = []
flux_left = []
flux_right = []
# with open('optimisation/derived_quantities.csv', 'r') as csvfile:
with open('optimisation/2_traps/type_B/beta=' +str(15)+ 'Kmin-1/derived_quantities.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 't(s)' not in row:
            if float(row[3]) > 300:
                T_exp.append(float(row[3]))
                flux_left.append(float(row[1]))
                flux_right.append(float(row[2]))

# plt.plot(T_exp, -np.asarray(flux_left))
# plt.plot(T_exp, -np.asarray(flux_right))
plt.plot(T_exp, -np.asarray(flux_left)-np.asarray(flux_right))

plt.show()
