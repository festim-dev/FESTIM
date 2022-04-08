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


plt.scatter(T_exp, d_exp, s=10)


def read_file(filename):
    out = []
    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            if 't(s)' in row:
                for e in row:
                    out.append([])
            else:
                if float(row[0]) > 450:
                    for i in range(len(row)):
                        out[i].append(float(row[i]))
    return out


ps = [[0.85, 1], [0.75, 3], [1, 3]]
fs = [0.05575002, 0.23397247, 0.22866025]
xys = [(530, 3e18), (320, 6e18), (600, 3e18)]
folder = 'spectra in plane E,n/derived_quantities/'


for i in range(3):
    out = read_file(folder + str(ps[i][0]) + '_' + str(ps[i][1]) + '.csv')
    t = out[0]
    T_sim = out[1]
    ret = out[2]
    flux = []
    for j in range(len(ret)-1):
        flux.append(-(ret[j+1] - ret[j])/(t[j+1] - t[j]))

    T_sim.pop(0)
    plt.plot(T_sim, flux, label='$E_1 = $' + '{:.2f}'.format(ps[i][0]) + ' eV' + '; $n_1 = $' + '{:.0e}'.format(ps[i][1]*1e-3) + ' at.fr')
    plt.annotate('f = {:.0e}'.format(fs[i]), xys[i])


plt.xlabel('T (K)')
plt.ylabel(r'Desorption flux (m$^{-2}$ s$^{-1}$)')
plt.minorticks_on()
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=0.7)
plt.legend()
plt.show()
