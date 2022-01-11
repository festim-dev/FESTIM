from .helpers import find_material_from_id, update_expressions, help_key, \
    parameters_helper

from .meshing import generate_mesh_from_vertices, read_subdomains_from_xdmf, \
                        mesh_and_refine, subdomains_1D, check_borders
from .initialising import read_from_xdmf, check_no_duplicates
from .formulations import formulation, formulation_extrinsic_traps, \
    define_variational_problem_heat_transfers

from .boundary_conditions import BoundaryCondition, DirichletBC, FluxBC, \
    define_dirichlet_bcs_T, apply_fluxes, \
    BoundaryConditionTheta, apply_boundary_conditions

from .solving import solve_it, solve_once, adaptive_stepsize

from .export import write_to_csv, export_txt, export_profiles, \
    define_xdmf_files, export_xdmf, treat_value, export_parameters
from .post_processing import run_post_processing, compute_error, \
    create_properties, calculate_maximum_volume, calculate_minimum_volume, \
    header_derived_quantities, derived_quantities, \
    check_keys_derived_quantities

from .generic_simulation import Simulation, run

import sympy as sp
x, y, z, t = sp.symbols('x[0] x[1] x[2] t')
R = 8.314462618  # Gas constant J.mol-1.K-1
k_B = 8.6173303e-5  # Boltzmann constant eV.K-1
