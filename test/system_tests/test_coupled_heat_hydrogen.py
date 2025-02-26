import numpy as np

import festim as F

test_mesh = F.Mesh1D(vertices=np.linspace(0, 1, 500))
test_mat = F.Material(D_0=1, E_D=0, thermal_conductivity=1)
test_subdomains = [F.VolumeSubdomain1D(id=1, borders=[0, 1], material=test_mat)]
test_H = F.Species("H", mobile=True)


def test_ensure_steady_T_field_passed_to_hydrogen_problem_the_same():
    """Test that the temperature field evaluated in the heat_problem is the same
    as that used in the temperature_fenics in the hydrogen problem"""

    T_func = lambda x: 2 * x[0] + 10
    test_vol_sub = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=test_mat)
    left_sub = F.SurfaceSubdomain1D(id=2, x=0)
    right_sub = F.SurfaceSubdomain1D(id=2, x=1)

    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedTemperatureBC(subdomain=left_sub, value=0),
            F.FixedTemperatureBC(subdomain=right_sub, value=T_func),
        ],
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=left_sub, value=1000, species=test_H),
            F.FixedConcentrationBC(subdomain=right_sub, value=0, species=test_H),
        ],
        species=[test_H],
        settings=F.Settings(atol=1, rtol=1e-10, transient=False),
    )

    test_coupled_problem = F.CoupledHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()
    test_coupled_problem.run()

    assert (
        test_coupled_problem.heat_problem.u
        == test_coupled_problem.hydrogen_problem.temperature_fenics
    )


def test_T_dependent_species():
    """Test that the function in the hydrogen problem which is temperature dependent
    has the correct value at x=1, when temperature field is space dependent"""

    T_func = lambda x: 2 * x[0] + 10
    c_func = lambda T: T

    test_vol_sub = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=test_mat)
    left_sub = F.SurfaceSubdomain1D(id=2, x=0)
    right_sub = F.SurfaceSubdomain1D(id=3, x=1)

    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedTemperatureBC(subdomain=left_sub, value=0),
            F.FixedTemperatureBC(subdomain=right_sub, value=T_func),
        ],
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=left_sub, value=0, species=test_H),
            F.FixedConcentrationBC(subdomain=right_sub, value=c_func, species=test_H),
        ],
        species=[test_H],
        settings=F.Settings(atol=1, rtol=1e-10, transient=False),
    )

    test_coupled_problem = F.CoupledHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()
    test_coupled_problem.run()

    assert np.isclose(test_coupled_problem.hydrogen_problem.u.x.array[-1], 12)


def test_transient_t_depedent_temp_and_T_dependent_species():
    """Test that the function in the hydrogen problem which is temperature dependent
    has the correct value at x=1, when temperature field is time and space dependent
    in transient"""

    T_func = lambda x, t: 2 * x[0] + 10 + t
    c_func = lambda T: T

    my_mat = F.Material(
        D_0=1, E_D=0, thermal_conductivity=1, density=1, heat_capacity=1
    )
    test_vol_sub = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=my_mat)
    left_sub = F.SurfaceSubdomain1D(id=2, x=0)
    right_sub = F.SurfaceSubdomain1D(id=3, x=1)

    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedTemperatureBC(subdomain=left_sub, value=0),
            F.FixedTemperatureBC(subdomain=right_sub, value=T_func),
        ],
        settings=F.Settings(
            atol=1e-10, rtol=1e-10, transient=True, stepsize=0.5, final_time=5
        ),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=left_sub, value=0, species=test_H),
            F.FixedConcentrationBC(subdomain=right_sub, value=c_func, species=test_H),
        ],
        species=[test_H],
        settings=F.Settings(
            atol=1, rtol=1e-10, transient=True, stepsize=0.5, final_time=5
        ),
    )

    test_coupled_problem = F.CoupledHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()
    test_coupled_problem.run()

    assert np.isclose(test_coupled_problem.hydrogen_problem.u.x.array[-1], 17)


def test_steady_non_matching_mesh():
    """Test that the function in the hydrogen problem which is temperature dependent
    has the correct value at x=1, when temperature field is time and space dependent,
    and the problems have mismatching meshes, in steady state"""

    T_func = lambda x: 3 * x[0] + 5
    c_func = lambda T: 10 * T

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
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh_2,
        subdomains=[test_vol_sub, left_sub, right_sub],
        boundary_conditions=[
            F.FixedConcentrationBC(subdomain=left_sub, value=0, species=test_H),
            F.FixedConcentrationBC(subdomain=right_sub, value=c_func, species=test_H),
        ],
        species=[test_H],
        settings=F.Settings(atol=1, rtol=1e-10, transient=False),
    )

    test_coupled_problem = F.CoupledHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()
    test_coupled_problem.run()

    assert np.isclose(test_coupled_problem.hydrogen_problem.u.x.array[-1], 80)


def test_transient_non_matching_mesh():
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

    test_coupled_problem = F.CoupledHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()
    test_coupled_problem.run()

    assert np.isclose(test_coupled_problem.hydrogen_problem.u.x.array[-1], 65)
