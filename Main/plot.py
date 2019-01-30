 ##This is an exemple of how to plot
 ## a basic graph with pyplot
 ## Rem

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv

T = []
d = []
t = []

with open('desorption1e22.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    d1 = []
    t1 = []
    T1 = []
    for row in plots:
        if 'd' not in row and 'T' not in row and 't' not in row:
            d1.append(float(row[0]))
            T1.append(float(row[1]))
            t1.append(float(row[2]))
    T.append(T1)
    d.append(d1)
    t.append(t1)

with open('desorption1e23.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    d1 = []
    t1 = []
    T1 = []
    for row in plots:
        if 'd' not in row and 'T' not in row and 't' not in row:
            d1.append(float(row[0]))
            T1.append(float(row[1]))
            t1.append(float(row[2]))
    T.append(T1)
    d.append(d1)
    t.append(t1)

with open('desorption.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    d1 = []
    t1 = []
    T1 = []
    for row in plots:
        if 'd' not in row and 'T' not in row and 't' not in row:
            d1.append(float(row[0]))
            T1.append(float(row[1]))
            t1.append(float(row[2]))
    T.append(T1)
    d.append(d1)
    t.append(t1)



plt.plot(T[0], d[0], label=r'fluence = $1\times 10^{22}$ D/m$ ^{2}$')
plt.plot(T[1], d[1], label=r'fluence = $1\times 10^{23}$ D/m$ ^{2}$')
plt.plot(T[2], d[2], label="desorption.csv")
plt.xlabel('T(K)')
plt.ylabel(r'desorption(D/m$ ^{2}$/s)')
plt.title('TDS')
plt.legend()
plt.show()
