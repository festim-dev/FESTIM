import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import rc


T = []
d = []

with open('experimental data/data.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if 't (s)' not in row:

            T.append(float(row[1]))
            d.append(float(row[2]))

T = np.array(T)
d = np.array(d)
sigma = 2e17
noise = np.random.normal(0, sigma, len(d))

d_noisy = d + noise

plt.plot(T, d_noisy, label=r"$\sigma = $" + str(sigma))
plt.legend()
plt.ylim(bottom=0)
plt.show()


with open('experimental data/noise_' + str(sigma) + '.csv', "w+") as f:
    writer = csv.writer(f, lineterminator='\n')
    for i in range(0, len(d_noisy)):
        writer.writerow([T[i], d_noisy[i]])
