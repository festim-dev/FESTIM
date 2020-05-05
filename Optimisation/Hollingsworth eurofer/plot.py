 ##This is an exemple of how to plot
 ## a basic graph with pyplot
 ## Rem

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv

try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
except:
    pass

flux = 3e17
implantation_time = 1e23/flux
resting_time = 100
ramp = 10/60
tds_time = (1000 - 300)/ramp


samples = ["S7", "S14", "S10"]
legends = ["0 dpa", "0.01 dpa", "0.1 dpa"]
colors = ["tab:blue", "tab:orange", "tab:green"]
for i in range(0, 3):
    T_exp = []
    d_exp = []
    with open(samples[i] + '/' + samples[i] + '.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=';')
        for row in plots:
            T_exp.append(float(row[0]))
            d_exp.append(float(row[1]))

    plt.scatter(T_exp, d_exp, s=10, color=colors[i])

    T_sim = []
    solute = []
    ret = []
    t = []
    trap1 = []
    with open(samples[i] + '/Results/derived_quantities/last_unpondered.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            if 't(s)' not in row:
                if float(row[0]) >= implantation_time + resting_time:
                    t.append(float(row[0]))
                    T_sim.append(float(row[1]))
                    ret.append(float(row[2]))
                    solute.append(float(row[3]))
                    trap1.append(float(row[4]))
    flux = []
    for j in range(len(ret)-1):
        flux.append(-(ret[j+1] - ret[j])/(t[j+1] - t[j]))
    T_sim.pop(0)
    plt.plot(T_sim, flux, alpha=1, color=colors[i], linestyle="dashed")

    T_sim = []
    solute = []
    ret = []
    t = []
    trap1 = []
    with open(samples[i] + '/Results/derived_quantities/last.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            if 't(s)' not in row:
                if float(row[0]) >= implantation_time + resting_time:
                    t.append(float(row[0]))
                    T_sim.append(float(row[1]))
                    ret.append(float(row[2]))
                    solute.append(float(row[3]))
                    trap1.append(float(row[4]))
    flux = []
    for j in range(len(ret)-1):
        flux.append(-(ret[j+1] - ret[j])/(t[j+1] - t[j]))
    T_sim.pop(0)
    plt.plot(T_sim, flux, label=legends[i], alpha=1, color=colors[i])

plt.minorticks_on()
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=0.7)
plt.xlim(left=300, right=900)
plt.xlabel(r'$T$ (K)')
plt.ylabel(r'Desorption flux (D m$^{-2}$ s $^{-1}$)')
# plt.title('TDS')
plt.legend()
plt.show()
