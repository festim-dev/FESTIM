from context import FESTIM
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
              "energy": 1.1,
              "density": 5e-4 * density,
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
                        "cells": 50,
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
                    ],
            }
            }
            }

    output = run(parameters, log_level=30)
    return output["derived_quantities"]


def mean_absolute_error(a, b, bounds=[], p=1):
    val = 0
    count = 0
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


def error(p):
    '''
    Compute average absolute error between simulation and reference
    '''
    print('-' * 40)
    global j
    j += 1
    print('i = ' + str(j))
    print('New simulation.')
    print('Point is:')
    print(p)
    for e in p:
        if e < 0:
            return 1e30
    res = simu(p)
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
    err = mean_absolute_error(interp_tds, ref)
    # err = RMSD(interp_tds, ref)
    err /= 1

    print('Average absolute error is :' + str(err) + ' ' + str(fatol) + str(xatol))
    # with open(folder + '/simulations_results.csv', 'a') as f:
    #     writer = csv.writer(f, lineterminator='\n', delimiter=',')
    #     writer.writerow([*p, err])
    return err


folder = '8e+16'
# folder = '2e+17'
ref = read_ref(folder + '/ref.csv')

if __name__ == "__main__":
    j = 0
    # real parameters are [0.87, 1.3e-3, 1.1, 0.5e-3]
    x0 = np.array([0.9, 1.6e-3, 1.2, 0.8e-3])
    # x0 = np.array([8.84009076e-01, 1.83210293e-03])#, 1.13631720e+00, 7.27940393e-04])
    x0 = np.array([0.9, 4.5e-3])#, 1.13631720e+00, 7.27940393e-04])
    # x0 = np.array([0.76948378, 1.3e-3])#, 1.13631720e+00, 7.27940393e-04])

    # gtol = 3e19
    # res = minimize(error, x0, method='BFGS',
    #                options={'disp': True, 'gtol': gtol})
    # print('Solution is: ' + str(res.x) + str(gtol))
    fatol = 1e15
    xatol = 1e-4
    res = minimize(error, x0, method='Nelder-Mead',
                   options={'disp': True, 'fatol': fatol, 'xatol': xatol})
    print('Solution is: ' + str(res.x))
