from FESTIM import *

x, y, z, t = sp.symbols('x[0] x[1] x[2] t')

implantation_time = 400
resting_time = 50
ramp = 8
delta_TDS = 500
TDS_time = int(delta_TDS / ramp) + 1
center = 4.5e-9  # + 20e-9
width = 2.5e-9
r = 0
flux = (1-r)*2.5e19 * (t <= implantation_time)
distribution = 1/(width*(2*3.14)**0.5) * \
    sp.exp(-0.5*((x-center)/width)**2)
xp = 1e-6
teta = (x < xp) * (x > 0) * (1/xp)

folder = 'Solution'

parameters = {
    "materials": [
        {
            "alpha": 1.1e-10,  # lattice constant ()
            "beta": 6*6.3e28,  # number of solute sites per atom (6 for W)
            "density": 6.3e28,
            "borders": [0, 20e-6],
            "E_diff": 0.39,
            "D_0": 4.1e-7,
            "id": 1
            }
            ],
    "traps": [
        {
            "energy": 0.87,
            "density": 1.3e-3*6.3e28,
            "materials": [1]
        },
        {
            "energy": 1.0,
            "density": 4e-4*6.3e28,
            "materials": [1]
        },
        {
            "energy": 1.5,
            "materials": [1],
            "density": 0,
            "type": 'extrinsic',
            "form_parameters":{
                "phi_0": flux,
                "n_amax": 1e-1*6.3e28,
                "f_a": distribution,
                "eta_a": 6e-4,
                "n_bmax": 1e-2*6.3e28,
                "f_b": teta,
                "eta_b": 2e-4,
            }
        }
        ],
    "mesh_parameters": {
            "initial_number_of_cells": 200,
            "size": 20e-6,
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
    "boundary_conditions": {
        "dc": [
            {
                "surface": [1],
                "value": 0
                },
            {
                "surface": [2],
                "value": 0
                }
        ],
        "solubility": [  # "surface", "S_0", "E_S", "pressure", "density"
            #{
            #    "surface": 1,
            #    "S_0": 1.3e-4,
            #    "E_S": 0.34,
            #    "pressure": 10*1e5,
            #    "density": 6.3e28
            #}
            ]
            },
    "temperature": {
            'type': "expression",
            'value': sp.printing.ccode(
                300 + (t > implantation_time+resting_time) *
                ramp * (t - (implantation_time+resting_time)))
        },
    "source_term": {
        'flux': sp.printing.ccode(flux),
        'distribution': sp.printing.ccode(distribution)
        },
    "solving_parameters": {
        "final_time": implantation_time +
        resting_time+TDS_time,
        "num_steps": 2*int(implantation_time +
                           resting_time+TDS_time),
        "adaptative_time_step": {
            "stepsize_change_ratio": 1.1,
            "t_stop": implantation_time + resting_time - 20,
            "stepsize_stop_max": 0.5,
            "dt_min": 1e-5
            },
        "newton_solver": {
            "absolute_tolerance": 1e10,
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
            "functions": ['solute', '1', '2', '3', '4', 'retention'],
            "labels":  ['solute', 'trap_1', 'trap_2',
                        'trap_3', 'trap_4', 'retention'],
            "folder": folder
        },
        "TDS": {
            "label": "desorption",
            "TDS_time": 450,
            "folder": folder
            }
        },

}

run(parameters)
