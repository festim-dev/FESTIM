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
d2 = []
d3 = []

with open('desorption1e22.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 'd' not in row and 'T' not in row and 't' not in row:
            d.append(float(row[0]))
            T.append(float(row[1]))
            t.append(float(row[2]))

with open('desorption.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 'd' not in row and 'T' not in row and 't' not in row:
            d2.append(float(row[0]))

with open('desorption1e23.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 'd' not in row and 'T' not in row and 't' not in row:
            d3.append(float(row[0]))

plt.plot(T, d, label=r'fluence = $1\times 10^{22}$ D/m$ ^{2}$')
plt.plot(T, d3, label=r'fluence = $1\times 10^{23}$ D/m$ ^{2}$')
plt.plot(T, d2, label="desorption.csv")
plt.xlabel('T(K)')
plt.ylabel(r'desorption(D/m$ ^{2}$/s)')
plt.title('TDS')
plt.legend()
plt.show()
