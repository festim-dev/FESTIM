from context2 import FESTIM
from FESTIM import *
from FESTIM.generic_simulation import run
import sympy as sp
import numpy as np
import csv
from scipy.interpolate import interp1d
from scipy.optimize import minimize

j = 0
implantation_time = 400
resting_time = 50
tds_time = 50


def read_ref(filename):
    '''
    Reads the data in filename
    '''
    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=';')
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
    size = 20e-6
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
               "density": p[1] * 1e-3 * density,
               "materials": 1,
            },
            {
              "energy": p[2],
              "density": p[3] * 1e-4 * density,
              "materials": 1,
            },
            {
                "energy": p[4],
                "materials": [1],
                "type": 'extrinsic',
                "form_parameters":{
                    "phi_0": 2.5e19 * (t <= 400),
                    "n_amax":  1e-1*6.3e28,
                    "f_a": distribution,
                    "eta_a": 6e-4,
                    "n_bmax": 1e-2*6.3e28,
                    "f_b": (x < 1e-6) * (x > 0) * (1/1e-6),
                    "eta_b": 2e-4,
                }
            }
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
                        "cells": 120,
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
                'value': 300 + (t > implantation_time + resting_time) * 8 * (t - (implantation_time + resting_time))
            },
        "source_term": {
            'value': 2.5e19 * (1 - r) * distribution * (t <= implantation_time)
            },
        "solving_parameters": {
            "final_time": implantation_time + resting_time + tds_time,
            "initial_stepsize": 0.1,
            "adaptive_stepsize": {
                "stepsize_change_ratio": 1.08,
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
            "derived_quantities": {
                "file": '_'.join(map(str, p)) + ".csv",
                "folder": folder + "/derived_quantities",
                "average_volume": [
                    {
                        "field": "T",
                        "volumes": [1]
                    }
                ],
                "total_volume": [
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
                    {
                        "field": "3",
                        "volumes": [1]
                    },
                    ],
            }
            }
            }

    output = run(parameters, log_level=30)
    return output["derived_quantities"]


def mean_absolute_error(a, b, bounds=[], p=1):
    val = 0
    count = 0
    coeff = 1
    for e in b:
        for b in bounds:
            if e[0] > b[0] and e[0] < b[1]:
                coeff = p
            else:
                coeff = 1
        val += coeff*abs(e[1] - a(e[0]))
        count += coeff
    val *= 1/count
    return val


def RMSD(a, b):
    val = 0
    for e in b:
        val += (e[1] - a(e[0]))**2
    val /= len(b)
    val = val**0.5
    return val


def error(xi):
    '''
    Compute average absolute error between simulation and reference
    '''
    print('-' * 40)
    global j
    j += 1
    print('i = ' + str(j))
    print('New simulation.')
    print('Point is:')
    print(xi)
    for e in xi:
        if e < 0:
            return 1e30
    res = simu(xi)
    res.pop(0)  # remove header
    res = np.array(res)
    # create d(ret)/dt
    T = []
    flux = []
    for i in range(0, len(res) - 1):
        if res[i][0] >= implantation_time + resting_time:
            T.append(res[i][1])
            flux.append(-(res[i+1][2] - res[i][2])/(res[i+1][0] - res[i][0]))
    T = np.array(T)
    flux = np.array(flux)
    interp_tds = interp1d(T, flux, fill_value='extrapolate')
    err = mean_absolute_error(interp_tds, ref, [], p=1)
    # err = RMSD(interp_tds, ref)
    err /= 1

    print('Average absolute error is :' + str(err) + ' ')# + str(fatol) + str(xatol))
    # with open(folder + '/simulations_results.csv', 'a') as f:
    #     writer = csv.writer(f, lineterminator='\n', delimiter=',')
    #     writer.writerow([*p, err])
    return err


folder = 'temp'
folder = 'optimisation_5D'
folder = 'cost_function_unpondered_average'
folder = 'optimisation_trap3'
folder = 'optimisation_5D_bis'

ref = read_ref('ref.csv')

if __name__ == "__main__":
    # real parameters are [0.87, 1.3e-3, 1, 4e-4]
    x0 = np.array([1, 1.4e-3])#, 1.2, 5e-4])
    x0 = np.array([0.85, 1.09, 0.96, 5.25, 1.37])
    # result is : [ 0.83581592  1.11364004  0.96148699  6.24247419  1.38985868] in 495 function evaluations

    # x0 = np.array([1.5, 1, 1.2])

    # result is : [ 1.41310529  4.82880627  8.69626235]

    # res = minimize(error, x0, method='BFGS', bounds=((0, None), (0, None)),
    #                options={'disp': True, 'gtol': 1e16})
    # print('Solution is: ' + str(res.x))

    res = minimize(error, x0, method='Nelder-Mead',
                   options={'disp': True, 'fatol': 1e15, 'xatol': 0.001})
    print('Solution is: ' + str(res.x))
