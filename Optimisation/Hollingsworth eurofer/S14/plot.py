 ##This is an exemple of how to plot
 ## a basic graph with pyplot
 ## Rem

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv


flux = 3e17
implantation_time = 1e23/flux
resting_time = 100
ramp = 10/60
tds_time = (1000 - 300)/ramp

T_exp = []
d_exp = []

with open('S14.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        T_exp.append(float(row[0]))
        d_exp.append(float(row[1]))


plt.scatter(T_exp, d_exp, label="exp", s=10)

T_sim = []
flux1, flux2 = [], []
solute = []
ret = []
t = []
trap1 = []
# trap2 = []
# trap3 = []


with open('Results/derived_quantities/last_unpondered.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 't(s)' not in row:
            if float(row[0]) >= implantation_time + resting_time:
                t.append(float(row[0]))
                # flux1.append(float(row[1]))
                # flux2.append(float(row[2]))
                T_sim.append(float(row[1]))

                ret.append(float(row[2]))
                solute.append(float(row[3]))
                trap1.append(float(row[4]))
                # trap2.append(float(row[7]))
                # trap3.append(float(row[8]))

fields = [ret, solute, trap1]#, trap2, trap3]
derivatives = [[] for i in range(len(fields))]
legends = ["Total", "Solute", "Trap 1"]#, "Trap 2", "Trap 3"]
for i in range(len(ret)-1):
    for j in range(0, len(derivatives)):
        derivatives[j].append(-(fields[j][i+1] - fields[j][i])/(t[i+1] - t[i]))
# plt.plot(T_sim, -np.asarray(flux1), label="flux1")
# plt.plot(T_sim, -np.asarray(flux2), label="flux2")
T_sim.pop(0)
for i in range(0, len(derivatives)):
    if i != 0:
        style = "dashed"
        width = 0.8
        plt.fill_between(T_sim, 0, derivatives[i], facecolor='grey', alpha=0.1)
    else:
        style = "-"
        width = 1.7
    if i != 1:
        plt.plot(T_sim, derivatives[i], linewidth=width, linestyle=style, label=legends[i], alpha=1)

# plt.xlim(300, 700)
plt.xlabel('T(K)')
plt.ylabel(r'desorption(D/m$ ^{2}$/s)')
plt.title('TDS')
plt.legend()
plt.show()
