 ##This is an exemple of how to plot
 ## a basic graph with pyplot
 ## Rem

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv

T_exp = []
d_exp = []

with open('ref.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        if 't(s)' not in row:

            T_exp.append(float(row[0]))
            d_exp.append(float(row[1]))


plt.scatter(T_exp, d_exp, s=10)


def read_file(filename):
    out = []
    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            if 't(s)' in row:
                for e in row:
                    out.append([])
            else:
                if float(row[0]) > 450:
                    for i in range(len(row)):
                        out[i].append(float(row[i]))
    return out


filenames = [
    'optimisation_1trap/derived_quantities/last.csv',
    'optimisation_4D/derived_quantities/last.csv',
    'optimisation_5D_correct_thickness/derived_quantities/last.csv'
]

linestyles = ['-.', '--', '-']
legends = ['1 trap', '2 traps', '3 traps']

for i in range(3):
    out = read_file(filenames[i])
    t = out[0]
    T_sim = out[1]
    ret = out[2]
    flux = []
    for j in range(len(ret)-1):
        flux.append(-(ret[j+1] - ret[j])/(t[j+1] - t[j]))

    T_sim.pop(0)
    plt.plot(
        T_sim, flux, color="tab:blue",
        linestyle=linestyles[i], label=legends[i])


plt.xlabel('T (K)')
plt.ylabel(r'Desorption flux (m$^{-2}$ s$^{-1}$)')
plt.minorticks_on()
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=0.7)
plt.legend()
# plt.plot(T_sim, derivatives[1])
plt.show()
