import numpy as np
import ufl

import festim as F

from .tools import error_L2

test_mesh = F.Mesh1D(vertices=np.linspace(0, 1, 500))
test_mat = F.Material(D_0=1, E_D=0, thermal_conductivity=1)
test_subdomains = [F.VolumeSubdomain1D(id=1, borders=[0, 1], material=test_mat)]
test_H = F.Species("H", mobile=True)


def test_MMS_coupled_problem():
    """Test that the function in the hydrogen problem which is temperature dependent
    has the correct value at x=1, when temperature field is time and space dependent
    in transient"""

    density = 1.2
    heat_capacity = 2.6
    thermal_conductivity = 4.2
    D_0 = 1.2
    E_D = 0.1
    k_0 = 2.2
    E_k = 0.2
    p_0 = 0.5
    E_p = 0.1
    n_trap = 5
    k_B = F.k_B

    final_time = 1

    test_mesh = F.Mesh1D(vertices=np.linspace(0, 1, 2000))

    test_mat = F.Material(
        D_0=D_0,
        E_D=E_D,
        thermal_conductivity=thermal_conductivity,
        density=density,
        heat_capacity=heat_capacity,
    )

    test_vol_sub = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=test_mat)
    left_sub = F.SurfaceSubdomain1D(id=2, x=0)
    right_sub = F.SurfaceSubdomain1D(id=3, x=1)

    test_mobile = F.Species("mobile", mobile=True)
    test_trapped = F.Species(name="trapped", mobile=False, subdomains=[test_vol_sub])
    test_traps = F.ImplicitSpecies(n=n_trap, others=[test_mobile, test_trapped])

    # define temperature sim
    def exact_T_solution(x, t):
        return 3 * x[0] ** 2 + 10 * t

    dTdt = 10
    mms_T_source = (
        test_mat.density * test_mat.heat_capacity * dTdt
        - test_mat.thermal_conductivity * 6
    )

    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedTemperatureBC(subdomain=left_sub, value=exact_T_solution),
            F.FixedTemperatureBC(subdomain=right_sub, value=exact_T_solution),
        ],
        sources=[F.HeatSource(value=mms_T_source, volume=test_vol_sub)],
        initial_condition=F.InitialTemperature(lambda x: exact_T_solution(x, 0)),
        settings=F.Settings(
            atol=1e-10,
            rtol=1e-10,
            transient=True,
            stepsize=final_time / 20,
            final_time=final_time,
        ),
    )

    # define hydrogen problem
    def exact_mobile_solution(x, t):
        return 2 * x[0] ** 2 + 15 * t

    def exact_trapped_solution(x, t):
        return 4 * x[0] ** 2 + 12 * t

    def exact_mobile_intial_cond(x):
        return 2 * x[0] ** 2

    def exact_trapped_intial_cond(x):
        return 4 * x[0] ** 2

    dmobiledt = 15
    dtrappeddt = 12

    def mms_mobile_source(x, t):
        return (
            dmobiledt
            - ufl.div(
                D_0
                * ufl.exp(-E_D / (k_B * (3 * x[0] ** 2 + 10 * t)))
                * ufl.grad(2 * x[0] ** 2 + 15 * t)
            )
            + k_0
            * ufl.exp(-E_k / (k_B * (3 * x[0] ** 2 + 10 * t)))
            * (2 * x[0] ** 2 + 15 * t)
            * (n_trap - (4 * x[0] ** 2 + 12 * t))
            - p_0
            * ufl.exp(-E_p / (k_B * (3 * x[0] ** 2 + 10 * t)))
            * (4 * x[0] ** 2 + 12 * t)
        )

    def mms_trapped_source(x, t):
        return (
            dtrappeddt
            - k_0
            * ufl.exp(-E_k / k_B * (3 * x[0] ** 2 + 10 * t))
            * (2 * x[0] ** 2 + 15 * t)
            * (n_trap - (4 * x[0] ** 2 + 12 * t))
            + p_0
            * ufl.exp(-E_p / k_B * (3 * x[0] ** 2 + 10 * t))
            * (4 * x[0] ** 2 + 12 * t)
        )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedConcentrationBC(
                subdomain=left_sub,
                value=exact_mobile_solution,
                species=test_mobile,
            ),
            F.FixedConcentrationBC(
                subdomain=right_sub,
                value=exact_mobile_solution,
                species=test_mobile,
            ),
            F.FixedConcentrationBC(
                subdomain=left_sub,
                value=exact_trapped_solution,
                species=test_trapped,
            ),
            F.FixedConcentrationBC(
                subdomain=right_sub,
                value=exact_trapped_solution,
                species=test_trapped,
            ),
        ],
        species=[test_mobile, test_trapped],
        reactions=[
            F.Reaction(
                reactant=[test_mobile, test_traps],
                product=test_trapped,
                k_0=1.0,
                E_k=0.2,
                p_0=0.1,
                E_p=0.3,
                volume=test_vol_sub,
            )
        ],
        initial_conditions=[
            F.InitialCondition(value=exact_mobile_intial_cond, species=test_mobile),
            F.InitialCondition(value=exact_trapped_intial_cond, species=test_trapped),
        ],
        sources=[
            F.ParticleSource(
                value=mms_mobile_source, volume=test_vol_sub, species=test_mobile
            ),
            F.ParticleSource(
                value=mms_trapped_source, volume=test_vol_sub, species=test_trapped
            ),
        ],
        settings=F.Settings(
            atol=1,
            rtol=1e-10,
            transient=True,
            stepsize=final_time / 20,
            final_time=final_time,
        ),
    )

    test_coupled_problem = F.CoupledtTransientHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()
    test_coupled_problem.run()

    T_computed = test_coupled_problem.heat_problem.u
    mobile_computed = test_mobile.post_processing_solution
    trapped_computed = test_trapped.post_processing_solution

    exact_T = lambda x: 3 * x[0] ** 2 + 10 * final_time
    exact_mobile = lambda x: 2 * x[0] ** 2 + 15 * final_time
    exact_trapped = lambda x: 4 * x[0] ** 2 + 12 * final_time

    L2_error_T = error_L2(T_computed, exact_T)
    L2_error_mobile = error_L2(mobile_computed, exact_mobile)
    L2_error_trapped = error_L2(trapped_computed, exact_trapped)

    assert L2_error_T < 2e-07
    assert L2_error_mobile < 2e-07
    assert L2_error_trapped < 2e-07


def test_coupled_problem_non_matching_mesh():
    """Test that the function in the hydrogen problem which is temperature dependent
    has the correct value at x=1, when temperature field is time and space dependent,
    and the problems have mismatching meshes, in transient"""

    T_func = lambda x, t: 3 * x[0] + 5 + t
    c_func = lambda T: 5 * T

    my_mat = F.Material(
        D_0=1, E_D=0, thermal_conductivity=1, density=1, heat_capacity=1
    )
    test_vol_sub = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left_sub = F.SurfaceSubdomain1D(id=2, x=0)
    right_sub = F.SurfaceSubdomain1D(id=3, x=1)
    test_mesh_2 = F.Mesh1D(vertices=np.linspace(0, 1, 1000))

    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedTemperatureBC(subdomain=left_sub, value=0),
            F.FixedTemperatureBC(subdomain=right_sub, value=T_func),
        ],
        settings=F.Settings(
            atol=1e-10, rtol=1e-10, transient=True, final_time=5, stepsize=0.5
        ),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh_2,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=left_sub, value=0, species=test_H),
            F.FixedConcentrationBC(subdomain=right_sub, value=c_func, species=test_H),
        ],
        species=[test_H],
        settings=F.Settings(
            atol=1, rtol=1e-10, transient=True, final_time=5, stepsize=0.5
        ),
    )

    test_coupled_problem = F.CoupledtTransientHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()
    test_coupled_problem.run()

    assert np.isclose(test_coupled_problem.hydrogen_problem.u.x.array[-1], 65)
