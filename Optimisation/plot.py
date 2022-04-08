 ##This is an exemple of how to plot
 ## a basic graph with pyplot
 ## Rem

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv

T_exp = []
d_exp = []

with open('2e+17/ref.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 't(s)' not in row:

            T_exp.append(float(row[0]))
            d_exp.append(float(row[1]))


plt.scatter(T_exp, d_exp, label="exp", s=10)

T_sim = []
ret = []
t = []
with open('2e+17/derived_quantities/0.872753866921_0.00132263650983_1.10346176556_0.00048996148925.csv', 'r') as csvfile:
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
plt.plot(T_sim, d, label="sim", linestyle='dashed', color='orange')

# plt.xlim(300, 700)
plt.xlabel('T(K)')
plt.ylabel(r'desorption(D/m$ ^{2}$/s)')
plt.title('TDS')
plt.legend()
plt.show()
