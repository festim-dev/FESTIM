import matplotlib.pyplot as plt

import numpy as np
import csv

filename = 'results/05_ITER_case_theta_sol2/derived_quantities.csv'

f = plt.figure()

t = []
inventory = []
solute = []
trap1 = []
trap2 = []
trap3 = []
trap4 = []
flux_coolant = []
flux_left = []
with open(filename, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')

    for row in plots:
        if 't(s)' in row:
            index_t = row.index('t(s)')
            index_ret_1 = row.index('Total retention volume 6')
            index_ret_2 = row.index('Total retention volume 7')
            index_ret_3 = row.index('Total retention volume 8')
            index_solute_1 = row.index('Total solute volume 6')
            index_solute_2 = row.index('Total solute volume 7')
            index_solute_3 = row.index('Total solute volume 8')
            index_trap_1 = row.index('Total 1 volume 8')
            index_trap_2 = row.index('Total 2 volume 8')
            index_trap_3 = row.index('Total 3 volume 7')
            index_trap_4 = row.index('Total 4 volume 6')
            index_flux_coolant = row.index('Flux surface 10: solute')
            index_flux_left = row.index('Flux surface 11: solute')

        else:
            t.append(float(row[index_t]))
            ret = float(row[index_ret_1])+float(row[index_ret_2])+float(row[index_ret_3])
            # ret *= 2*12e-3  # whole monoblock
            # ret *= 1/6.022e23  # in mol
            # ret *= 1  # in g
            inventory.append(ret)
            solute.append(float(row[index_solute_1])+float(row[index_solute_2])+float(row[index_solute_3]))
            trap1.append(float(row[index_trap_1]))
            trap2.append(float(row[index_trap_2]))
            trap3.append(float(row[index_trap_3]))
            trap4.append(float(row[index_trap_4]))
            flux_left_val = float(row[index_flux_left])
            # flux_left_val *= 2*12e-3  # whole monoblock
            # flux_left_val *= 1/6.022e23  # in mol
            flux_left.append(flux_left_val)
            flux_coolant.append(float(row[index_flux_coolant]))


# # Plot retention

plt.xlabel('t (s)')
plt.ylabel('Integrated retention (H)')
plt.plot(t, inventory, label=r'Inventory', linewidth=1.5)
plt.plot(t, solute, label=r'Solute', linestyle='--')
plt.plot(t, trap1, label=r'Trap 1', linestyle='--')
plt.plot(t, trap2, label=r'Trap 2', linestyle='--')
plt.plot(t, trap3, label=r'Trap 3', linestyle='--')
plt.plot(t, trap4, label=r'Trap 4', linestyle='--')
plt.minorticks_on()
plt.grid(which='minor', alpha=0.3)
plt.grid(which='major', alpha=0.7)
plt.legend()
plt.show()

# # # Plot flux at coolant

# plt.plot(t, flux_coolant, label="Flux coolant")
# plt.show()

# # # Plot left flux

# plt.plot(t, flux_left, label="Flux left")
# plt.legend()
# plt.show()
