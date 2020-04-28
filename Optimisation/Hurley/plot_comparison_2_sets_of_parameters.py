import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv


T_exp = []
flux_left = []
flux_right = []
with open('optimisation/1_trap/derived_quantities.csv', 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                if 't(s)' not in row:
                    if float(row[3]) > 300:
                        T_exp.append(float(row[3]))
                        flux_left.append(float(row[1]))
                        flux_right.append(float(row[2]))

plt.plot(T_exp, -np.asarray(flux_left)-np.asarray(flux_right), label="1 trapping site", alpha=0.7)

T_exp = []
flux_left = []
flux_right = []
with open('optimisation/2_traps/type_B/beta=2Kmin-1/derived_quantities.csv', 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                if 't(s)' not in row:
                    if float(row[3]) > 300:
                        T_exp.append(float(row[3]))
                        flux_left.append(float(row[1]))
                        flux_right.append(float(row[2]))
plt.plot(T_exp, -np.asarray(flux_left)-np.asarray(flux_right), label="2 trapping sites", alpha=0.7)

T_exp = []
flux_left = []
flux_right = []
with open('optimisation/3_traps/type_B/beta=2Kmin-1/derived_quantities.csv', 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                if 't(s)' not in row:
                    if float(row[3]) > 300:
                        T_exp.append(float(row[3]))
                        flux_left.append(float(row[1]))
                        flux_right.append(float(row[2]))
plt.plot(T_exp, -np.asarray(flux_left)-np.asarray(flux_right), label="3 trapping sites", alpha=0.7)
plt.xlabel(r"T (K)")
plt.ylabel(r"Desorption flux (m$^{-2}$ s$^{-1}$)")
plt.legend()
plt.minorticks_on()
plt.xlim(right=440)
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=0.7)
plt.show()
