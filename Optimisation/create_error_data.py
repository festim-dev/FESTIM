from optimisation_TDS import *

E_bounds = [0.87, 0.87]
n_bounds = [1.3e-3, 1.3e-3]


E_range = np.linspace(E_bounds[0], E_bounds[1], 1)
n_range = np.linspace(n_bounds[0], n_bounds[1], 1)
print(len(E_range), len(n_range))

e = []
for E in E_range:
    for n in n_range:
        err = error([E, n, 1.1, 0.5e-3])
        e.append(err)
        with open(folder + '/cost_function_data.csv', 'a+') as f:
            writer = csv.writer(f, lineterminator='\n', delimiter=',')
            writer.writerow([E, n, err])
