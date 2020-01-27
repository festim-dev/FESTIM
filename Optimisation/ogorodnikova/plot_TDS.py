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


plt.scatter(T_exp, d_exp, label="exp", s=10)

# T_sim = []
# ret = []
# t = []
# with open('cost_function_non_pondered_average/derived_quantities/0.86_0.0015.csv', 'r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         if 't(s)' not in row:
#             if float(row[0]) > 450:
#                 t.append(float(row[0]))
#                 T_sim.append(float(row[1]))
#                 ret.append(float(row[2]))

# d = []
# for i in range(len(ret)-1):
#     d.append(-(ret[i+1] - ret[i])/(t[i+1] - t[i]))
# T_sim.pop(0)
# plt.plot(T_sim, d, label="sim unpondered", linestyle='dashed', color='orange')


T_sim = []
ret = []
t = []
with open('optimisation_5D/derived_quantities/0.849199704993_1.08753443745_0.967984862256_5.24800051466_1.37441734167.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 't(s)' not in row:
            if float(row[0]) > 450:
                t.append(float(row[0]))
                T_sim.append(float(row[1]))
                ret.append(float(row[2]))

d = []
for i in range(len(ret)-1):
    d.append(-(ret[i+1] - ret[i])/(t[i+1] - t[i]))
T_sim.pop(0)
plt.plot(T_sim, d, label="sim pondered", linestyle='dashed', color='grey')

# plt.xlim(300, 700)
plt.xlabel('T(K)')
plt.ylabel(r'desorption(D/m$ ^{2}$/s)')
plt.minorticks_on()
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=0.7)
# plt.title('TDS')
plt.legend()
plt.show()

## Plot TDS with each trap
T_sim = []
ret = []
trap1 = []
trap2 = []
trap3 = []
solute = []
t = []
with open('optimisation_5D/derived_quantities/0.849199704993_1.08753443745_0.967984862256_5.24800051466_1.37441734167.csv', 'r') as csvfile:
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
legends = ["Total", "Solute", "Trap 1", "Trap 2", "Trap 3"]
print(derivatives)
for i in range(len(ret)-1):
    for j in range(0, len(derivatives)):
        derivatives[j].append(-(fields[j][i+1] - fields[j][i])/(t[i+1] - t[i]))

T_sim.pop(0)
for i in range(0, len(derivatives)):
    if i != 0:
        style = "dashed"
        plt.fill_between(T_sim, 0, derivatives[i], facecolor='grey', alpha=0.3)
    else:
        style = "-"
    plt.plot(T_sim, derivatives[i], linestyle=style, label=legends[i])
plt.scatter(T_exp, d_exp, label="exp", s=10)
plt.xlabel('T(K)')
plt.ylabel(r'time derivative (m$^{-2}$ s$^{-1}$)')
plt.minorticks_on()
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=0.7)
plt.legend()
# plt.plot(T_sim, derivatives[1])
plt.show()
