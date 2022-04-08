from optimisation_TDS import *
import sys

E_bounds = [0.5, 1.2]
n_bounds = [0.1e-3, 6e-3]


E_range = np.linspace(E_bounds[0], E_bounds[1], 9*3) #[0.773, 0.742] #
n_range = np.linspace(n_bounds[0], n_bounds[1], 11*2) # [5.72e-3] # 
e = []
for E in E_range:
    if E > 0.72 and E < 0.75:
        for n in n_range:
            if n > 3.2e-3 and n < 3.7e-3:
                try:
                    err = error([E, n])
                    e.append(err)
                    busy = True
                    while busy:
                        try:
                            with open(folder + '/cost_function_data_very_refined.csv', 'a+') as f:
                                writer = csv.writer(f, lineterminator='\n', delimiter=',')
                                writer.writerow([E, n, err])
                            busy = False
                        except KeyboardInterrupt:
                            break
                except KeyboardInterrupt:
                    break
                except:
                    pass
