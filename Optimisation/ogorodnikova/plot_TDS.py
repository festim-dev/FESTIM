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


T_exp = []
d_exp = []

with open('ref.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        if 't(s)' not in row:

            T_exp.append(float(row[0]))
            d_exp.append(float(row[1]))


plt.scatter(T_exp, d_exp, label="Exp", s=10)

## Plot TDS with each trap
T_sim = []
ret = []
trap1 = []
trap2 = []
trap3 = []
solute = []
t = []
with open('optimisation_5D_correct_thickness/derived_quantities/last.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 't(s)' not in row:
            if float(row[0]) > 450:
                t.append(float(row[0]))
                T_sim.append(float(row[1]))
                ret.append(float(row[2]))
                solute.append(float(row[3]))
                trap1.append(float(row[4]))
                trap2.append(float(row[5]))
                trap3.append(float(row[6]))

fields = [ret, solute, trap1, trap2, trap3]
derivatives = [[] for i in range(len(fields))]
legends = ["Total", "Solute", "", "", ""]
print(derivatives)
for i in range(len(ret)-1):
    for j in range(0, len(derivatives)):
        derivatives[j].append(-(fields[j][i+1] - fields[j][i])/(t[i+1] - t[i]))

T_sim.pop(0)
for i in range(0, len(derivatives)):
    if i != 0:
        style = "dashed"
        color = "grey"
        width = 0.8
        plt.fill_between(T_sim, 0, derivatives[i], facecolor='grey', alpha=0.1)
    else:
        style = "-"
        color = "tab:blue"
        width = 1.7
    if i != 1:
        plt.plot(T_sim, derivatives[i], linewidth=width, linestyle=style, label=legends[i], alpha=1, color=color)


plt.xlabel(r'$T$ (K)')
plt.ylabel(r'Desorption flux (m$^{-2}$ s$^{-1}$)')
plt.minorticks_on()
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=0.7)
# plt.legend()
# plt.plot(T_sim, derivatives[1])
plt.show()
