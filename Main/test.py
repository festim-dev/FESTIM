import FESTIM
import fenics
import pytest
import sympy as sp


def test_mesh_and_refine_meets_refinement_conditions():
    def create_subdomains(x1, x2):
        class domain(FESTIM.SubDomain):
            def inside(self, x, on_boundary):
                return x[0] >= x1 and x[0] <= x2
        domain = domain()
        return domain

    def mesh_parameters(ini, size, refs, pos):
        param = {
            "initial_number_of_cells": ini,
            "size": size,
            "refinements":  []
        }
        for i in range(len(refs)):
            param["refinements"].append({"cells": refs[i], "x": pos[i]})
        return param
    refinements = [[[2, 3], [0.5, 0.25]], [[3, 11], [0.5, 0.25]]]
    for i in range(len(refinements)):
        param = mesh_parameters(2, 1, refinements[i][0], refinements[i][1])
        mesh = FESTIM.mesh_and_refine(param)

        mf1 = FESTIM.MeshFunction('size_t', mesh, 1)
        mf2 = FESTIM.MeshFunction('size_t', mesh, 1)
        subdomain1 = create_subdomains(0, refinements[i][1][1])
        subdomain1.mark(mf1, 1)
        subdomain2 = create_subdomains(0, refinements[i][1][0])
        subdomain2.mark(mf2, 2)
        nb_cell_1 = 0
        nb_cell_2 = 0
        for cell in FESTIM.cells(mesh):
            cell_no = cell.index()
            if mf1.array()[cell_no] == 1:
                nb_cell_1 += 1
            if mf2.array()[cell_no] == 2:
                nb_cell_2 += 1
        assert nb_cell_1 >= refinements[i][0][1]
        assert nb_cell_2 >= refinements[i][0][0]


def test_run_MMS():
    '''
    Test function run() for several refinements
    '''

    u = 1 + sp.exp(-4*fenics.pi**2*FESTIM.t)*sp.cos(2*fenics.pi*FESTIM.x)
    v = 1 + sp.exp(-4*fenics.pi**2*FESTIM.t)*sp.cos(2*fenics.pi*FESTIM.x)

    def parameters(h, dt, final_time, u, v):
        size = 1
        folder = 'Solution_Test'
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
        D = D_0 * fenics.exp(-E_diff/k_B/T)
        v_i = v_0 * fenics.exp(-E_t/k_B/T)
        v_m = D/alpha/alpha/beta

        f = sp.diff(u, FESTIM.t) + sp.diff(v, FESTIM.t) - \
            D * sp.diff(u, FESTIM.x, 2)
        g = sp.diff(v, FESTIM.t) + v_i*v - v_m * u * (n_trap-v)
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
                        "value": u,
                        "component": 0
                    },
                    {
                        "surface": [1, 2],
                        "value": v,
                        "component": 1
                    }
                ],
                "solubility": [  # "surface", "S_0", "E_S", "pressure", "density"
                ]
                },
            "temperature": {
                    'type': "expression",
                    'value': T
                },
            "source_term": {
                'flux': f,
                'distribution': 1
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
                    "folder": folder
                },
                "xdmf": {
                    "functions": [],
                    "labels":  [],
                    "folder": folder
                },
                "TDS": {
                    "file": "desorption",
                    "TDS_time": 0,
                    "folder": folder
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
        output = FESTIM.run(parameters(h, dt, final_time, u, v))
        error_max_u = output["error"][0][1]
        error_max_v = output["error"][0][2]
        msg = 'Maximum error on u is:' + str(error_max_u) + '\n \
            Maximum error on v is:' + str(error_max_v) + '\n \
            with h = ' + str(h) + '\n \
            with dt = ' + str(dt)
        print(msg)
        assert error_max_u < tol_u and error_max_v < tol_v
