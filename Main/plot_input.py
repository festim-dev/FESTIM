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

source = input('Enter the file name: ')
with open(source, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 'd' not in row and 'T' not in row and 't' not in row:
            d.append(float(row[0]))
            T.append(float(row[1]))
            t.append(float(row[2]))

plt.plot(T, d, label=source)
plt.xlabel('T(K)')
plt.ylabel(r'desorption(D/m$ ^{2}$/s)')
plt.title('TDS')
plt.legend()
plt.show()
