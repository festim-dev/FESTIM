from context import FESTIM
from FESTIM import *
from FESTIM.generic_simulation import run
import sympy as sp
import numpy as np
import csv
from scipy.interpolate import interp1d
from scipy.optimize import minimize

i = 0
implantation_time = 400
resting_time = 50
tds_time = 100


def read_ref(filename):
    '''
    Reads the data in filename
    '''
    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        res = []
        for row in plots:
            if 'd' not in row and 'T' not in row and 't (s)' not in row:
                res.append([float(row[i]) for i in [0, 1]])

    return res


def simu(p):
    '''
    Runs the simulation with parameters p
    '''
    file_name = "desorption-"
    for e in p:
        file_name += ';' + str(e)

    density = 6.3e28
    size = 2e-5
    r = 0
    center = 4.5e-9
    width = 2.5e-9
    distribution = 1/(width*(2*3.14)**0.5) * sp.exp(-0.5*((x-center)/width)**2)
    parameters = {
        "materials": [
            {
                "alpha": 1.1e-10,  # lattice constant ()
                "beta": 6 * density,  # number of solute sites per atom (6 for W)
                "borders": [0, size],
                "E_diff": 0.39,
                "D_0": 4.1e-7,
                "id": 1
                }
                ],
        "traps": [

            {
               "energy": p[0],
               "density": p[1] * density,
               "materials": 1,
            },
            {
              "energy": p[2],
              "density": p[3] * density,
              "materials": 1,
            },
            ],
        "mesh_parameters": {
                "initial_number_of_cells": 200,
                "size": size,
                "refinements": [
                    {
                        "cells": 300,
                        "x": 3e-6
                    },
                    {
                        "cells": 100,
                        "x": 30e-9
                    }
                ],
            },
        "boundary_conditions": [
                {
                    "surface": [1, 2],
                    "value": 0,
                    "component": 0,
                    "type": "dc"
                }
            ],
        "temperature": {
                'type': "expression",
                'value': 300 + (t > implantation_time + resting_time) * 4 * (t - (implantation_time + resting_time))
            },
        "source_term": {
            'value': 2.5e19 * (1 - r) * distribution * (t <= implantation_time)
            },
        "solving_parameters": {
            "final_time": implantation_time + resting_time + tds_time,
            "initial_stepsize": 0.5,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1.1,
                "t_stop": implantation_time + resting_time - 20,
                "stepsize_stop_max": 0.4,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_iterations": 50,
            }
            },
        "exports": {
            "TDS": {
                "file": file_name,
                "TDS_time": implantation_time + resting_time,
                "folder": folder + "/TDS profiles"
                },
            "derived_quantities": {
                "file": "derived_quantities.csv",
                "folder": "derived_quantities",
                "total_volume": [
                    {
                        "field": "temperature",
                        "volumes": [1]
                    },
                    {
                        "field": "retention",
                        "volumes": [1]
                    },
                    {
                        "field": "solute",
                        "volumes": [1]
                    },
                    {
                        "field": "1",
                        "volumes": [1]
                    },
                    {
                        "field": "2",
                        "volumes": [1]
                    },
                    ],
            }
            }
            }

    output = run(parameters, log_level=30)
    return output["derived_quantities"]


def error(p):
    '''
    Compute average absolute error between simulation and reference
    '''
    print('-' * 40)
    print('New simulation.')
    print('Point is:')
    print(p)
    res = simu(p)
    res.pop(0)  # remove header
    res = np.array(res)
    # create d(ret)/dt
    tds = np.array()
    for i in range(0, len(res)):
        tds.append([res[i][0], -(res[i+1][1] - res[i][1])/(res[i+1][0] - res[i][0])])

    interp_tds = interp1d(
        tds[implantation_time + resting_time:, 1],
        tds[implantation_time + resting_time:, 2],
        fill_value='extrapolate')
    err = 0
    for e in ref:
        T = e[0]
        err += abs(e[1] - interp_tds(T))
    err *= 1/len(ref)
    print('Average absolute error is :' + str(err))
    with open(folder + '/simulations_results.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n', delimiter=',')
        writer.writerow([*p, err])

    return err
folder = '8e+16'
folder = '2e+17'
ref = read_ref(folder + '/ref.csv')


x0 = np.array([0.86931, 1.4929e-3, 1.105, 0.5856860e-3])
res = minimize(error, x0, method='Nelder-Mead',
               options={'disp': True, 'ftol': 0.001, 'xtol': 0.001})
print('Solution is: ' + str(res.x))
