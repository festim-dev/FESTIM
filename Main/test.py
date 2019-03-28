from FESTIM import *


def test_run():
    '''
    Test function run() for several refinements
    '''
    x, y, z, t = sp.symbols('x[0] x[1] x[2] t')
    u = 1 + sp.exp(-4*pi**2*t)*sp.cos(2*pi*x)
    v = 1 + sp.exp(-4*pi**2*t)*sp.cos(2*pi*x)

    def parameters(h, dt, final_time, u, v):
        size = 1

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

        f = sp.diff(u, t) + sp.diff(v, t) - D * sp.diff(u, x, 2)
        g = sp.diff(v, t) + v_i*v - v_m * u * (n_trap-v)
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
                    "value": u,
                    "component": 0
                },
                {
                    "value": v,
                    "component": 1
                }
            ],

            "mesh_parameters": {
                    "initial_number_of_cells": round(size/h),
                    "size": size,
                    "refinements": [
                    ],
                },
            "boundary_conditions": {
                "dc": [
                    {
                        "surface": [1, 2],
                        "value": sp.printing.ccode(u),
                        "component": 0
                    },
                    {
                        "surface": [1, 2],
                        "value": sp.printing.ccode(v),
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
                "final_time": final_time,
                "num_steps": round(1/dt),
                "adaptative_time_step": {
                    "stepsize_change_ratio": 1,
                    "t_stop": 0,
                    "stepsize_stop_max": dt,
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
                    "functions": [],
                    "times": [],
                    "labels": [],
                    "folder": ''
                },
                "xdmf": {
                    "functions": [],
                    "labels":  [],
                    "folder": ''
                },
                "TDS": {
                    "label": "desorption",
                    "TDS_time": 450,
                    "folder": ''
                    },
                "error": [
                    {
                        "exact_solution": [u, v],
                        "norm": 'error_max',
                        "degree": 4
                    }
                ]
                },
        }
        return parameters

    tol_u = 1e-7
    tol_v = 1e-1
    sizes = [1/1600, 1/1700]
    dt = 1/50
    final_time = 0.1
    for h in sizes:
        output = run(parameters(h, dt, final_time, u, v))
        error_max_u = output["error"][0][1]
        error_max_v = output["error"][0][2]
        msg = 'Maximum error on u is:' + str(error_max_u) + '\n \
            Maximum error on v is:' + str(error_max_v) + '\n \
            with h = ' + str(h) + '\n \
            with dt = ' + str(dt)
        print(msg)
        assert error_max_u < tol_u and error_max_v < tol_v
