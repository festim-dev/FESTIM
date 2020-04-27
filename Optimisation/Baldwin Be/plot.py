import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv


t_exp = []
d_exp = []

with open('ref.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        t_exp.append(float(row[0]))
        d_exp.append(float(row[1]))

plt.scatter(t_exp, d_exp, s=10, zorder=3)

solute = []
trap1 = []
trap2 = []
ret = []
t = []
surf = []

with open('Results/derived_quantities/last_recomb.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 't(s)' not in row:

            if float(row[0]) > 306:
                t.append(float(row[0]))
                solute.append(float(row[1]))
                trap1.append(float(row[2]))
                trap2.append(float(row[3]))
                ret.append(float(row[4]))

fields = [ret, trap1, trap2]
derivatives = [[] for i in range(len(fields))]
legends = ["ret", "Trap 1", "Trap 2"]
for i in range(len(solute)-1):
    for j in range(0, len(derivatives)):
        derivatives[j].append(-(fields[j][i+1] - fields[j][i])/(t[i+1] - t[i]))


t.pop(0)

for i in range(0, len(derivatives)):
    if i != 0:
        color = "grey"
        style = "dashed"
        width = 0.8
        plt.fill_between(t, 0, derivatives[i], facecolor='grey', alpha=0.1)
    else:
        color = "tab:blue"
        style = "-"
        width = 1.7
    # if i != 1:
    plt.plot(
        t, derivatives[i],
        linewidth=width, linestyle=style, color=color,
        label=legends[i], alpha=1)
# plt.xlim(300, 700)
plt.xlabel('t (s)')
plt.ylabel(r'Desorption flux (D m$^{-2}$ s $^{-1}$)')
# plt.title('TDS')
plt.minorticks_on()
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=0.7)
# plt.legend()
plt.show()
