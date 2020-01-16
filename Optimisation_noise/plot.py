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

with open('experimental data/data.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 't (s)' not in row:

            T.append(float(row[1]))
            d.append(float(row[2]))


plt.plot(T, d, label="exp", linestyle='dashed')


plt.xlabel('T(K)')
plt.ylabel(r'desorption(D/m$ ^{2}$/s)')
plt.title('TDS')
plt.legend()
plt.show()
