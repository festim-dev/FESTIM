from FESTIM import *

x, y, z, t = sp.symbols('x[0] x[1] x[2] t')

size = 1
sizing = (1/size)**2

u = 1 + sizing*x**2 + sp.sin(t)
v = 1 + sizing*x**2 + sp.cos(t)
v_0 = 1e13
E_t = 1.5
T = 700
density = 1 * 6.3e28
beta = 6*density
alpha = 1.1e-10
n_trap = 1e-1*density
E_diff = 0.39
D_0 = 4.1e-7
k_B = 8.6e-5
D = D_0 * exp(-E_diff/k_B/T)
v_i = v_0 * exp(-E_t/k_B/T)
v_m = D/alpha/alpha/beta

f = sp.cos(t) - 2*D - sp.sin(t)
g = v_i*v - v_m * u * (n_trap-v)-sp.sin(t)


folder = 'SolutionTestMMS/'


def parameters(cells, num_steps):
    parameters = {
        "materials": [
            {
                "alpha": alpha,  # lattice constant ()
                "beta": beta,  # number of solute sites per atom (6 for W)
                "density": density,
                "borders": [0, size],
                "E_diff": E_diff,
                "D_0": D_0,
                "id": 1
                }
                ],
        "traps": [
            {
                "energy": E_t,
                "density": n_trap,
                "materials": 1,
                "source_term": g
            }
            ],
        "initial_conditions": [
            {
                "value": 1 + sizing*x**2 + sp.sin(t),
                "component": 0
            },
            {
                "value": 1 + sizing*x**2 + sp.cos(t),
                "component": 1
            }
        ],

        "mesh_parameters": {
                "initial_number_of_cells": cells,
                "size": size,
                "refinements": [
                ],
            },
        "boundary_conditions": {
            "dc": [
                {
                    "surface": [1, 2],
                    "value": sp.printing.ccode(1 + sizing*x**2 + sp.sin(t)),
                    "component": 0
                },
                {
                    "surface": [1, 2],
                    "value": sp.printing.ccode(1 + sizing*x**2 + sp.cos(t)),
                    "component": 1
                }
            ],
            "solubility": [  # "surface", "S_0", "E_S", "pressure", "density"
            ]
            },
        "temperature": {
                'type': "expression",
                'value': sp.printing.ccode(T)
            },
        "source_term": {
            'flux': sp.printing.ccode(f),
            'distribution': sp.printing.ccode(1)
            },
        "solving_parameters": {
            "final_time": 1,
            "num_steps": num_steps,
            "adaptative_time_step": {
                "stepsize_change_ratio": 1,
                "t_stop": 0,
                "stepsize_stop_max": 0.5,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-9,
                "maximum_it": 50,
            }
            },
        "exports": {
            "txt": {
                "functions": ['retention'],
                "times": [100],
                "labels": ['retention'],
                "folder": folder
            },
            "xdmf": {
                "functions": ['solute', '1'],
                "labels":  ['u', 'v'],
                "folder": folder
            },
            "TDS": {
                "label": "desorption",
                "TDS_time": 450,
                "folder": folder
                },
            "error": [
                {
                    "exact_solution": [u, v],
                    "norm": 'L2',
                    "degree": 4
                }
            ]
            },
    }
    return parameters

h = [1.0, 0.5, 0.3, 0.1, 0.05, 0.016666666666666666, 0.01, 0.005, 0.0008333333333333334]
nb_cells = [1, 2, 3, 4, 5, 6, 10, 20, 30, 60, 100, 200, 600]
error_h = []
for nb_cell in nb_cells:
    param = parameters(nb_cell, 50)
    print('Running FESTIM MMS with ' + str(nb_cell) + ' cells')
    output = run(param)
    error_1 = [i[0] for i in output['error']]
    error_2 = [i[1] for i in output['error']]
    print('Maximum Error on u :' + str(max(error_1)))
    print('Maximum Error on v :' + str(max(error_2)))

    average_1 = sum(error_1)/len(error_1)
    average_2 = sum(error_2)/len(error_2)
    error_h.append(['h', size/nb_cell, average_1, average_2])

nb_steps = [10, 20, 30, 50, 100, 200, 600]
error_dt = []
for num in nb_steps:
    param = parameters(60, num)
    print('Running FESTIM MMS with ' + str(num) + ' time steps')
    output = run(param)
    error_1 = [i[0] for i in output['error']]
    error_2 = [i[1] for i in output['error']]
    print('Maximum Error on u :' + str(max(error_1)))
    print('Maximum Error on v :' + str(max(error_2)))

    average_1 = sum(error_1)/len(error_1)
    average_2 = sum(error_2)/len(error_2)
    error_dt.append(['dt', 1/num, average_1, average_2])
busy = True
while busy is True:
    try:
        with open('convergence_rates.csv', "w+") as f:
            busy = False
            writer = csv.writer(f, lineterminator='\n')
            for val in error_h:
                writer.writerows([val])
            for val in error_dt:
                writer.writerows([val])
        busy = False
    except:
        input('Busy')
