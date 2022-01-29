import fenics
import FESTIM


def test_fluxes_chemical_pot():
    '''
    This test that the function boundary_conditions.create_H_fluxes()
    returns the correct formulation in the case of conservation
    of chemical potential
    '''

    Kr_0 = 2
    E_Kr = 3
    S_0 = 2
    E_S = 3
    order = 2
    k_B = FESTIM.k_B

    mesh = fenics.UnitIntervalMesh(10)
    my_sim = FESTIM.Simulation({"traps": []})
    my_sim.boundary_conditions = [
        FESTIM.RecombinationFlux(Kr_0=Kr_0, E_Kr=E_Kr, order=order, surfaces=1),
        FESTIM.FluxBC(value=2*FESTIM.x + FESTIM.t, surfaces=[1, 2]),
    ]
    my_sim.mesh = FESTIM.Mesh(mesh=mesh)
    my_sim.define_function_spaces()
    my_sim.initialise_concentrations()

    my_sim.settings.chemical_pot = True
    my_sim.ds = fenics.ds
    my_sim.T = FESTIM.Temperature("expression", value=1000)
    my_sim.T.create_functions(my_sim.V_CG1)

    S = S_0*fenics.exp(-E_S/k_B/my_sim.T.T)

    my_sim.S = S
    my_sim.F = 0
    my_sim.create_H_fluxes()
    expressions = my_sim.expressions
    test_sol = my_sim.v
    sol = my_sim.u
    Kr_0 = expressions[0]
    E_Kr = expressions[1]
    Kr = Kr_0 * fenics.exp(-E_Kr/k_B/my_sim.T.T)
    expected_form = 0
    expected_form += -test_sol * (-Kr*(sol*S)**order)*fenics.ds(1)
    expected_form += -test_sol*expressions[2]*fenics.ds(1)
    expected_form += -test_sol*expressions[2]*fenics.ds(2)
    assert expected_form.equals(my_sim.F)


def test_fluxes():
    Kr_0 = 2
    E_Kr = 3
    order = 2
    k_B = FESTIM.k_B

    mesh = fenics.UnitIntervalMesh(10)
    my_sim = FESTIM.Simulation({"traps": []})
    my_sim.boundary_conditions = [
        FESTIM.RecombinationFlux(Kr_0=Kr_0, E_Kr=E_Kr, order=order, surfaces=1),
        FESTIM.FluxBC(value=2*FESTIM.x + FESTIM.t, surfaces=[1, 2]),
    ]
    my_sim.mesh = FESTIM.Mesh(mesh=mesh)
    my_sim.define_function_spaces()
    my_sim.F = 0
    my_sim.initialise_concentrations()
    my_sim.ds = fenics.ds
    my_sim.T = FESTIM.Temperature("expression", value=1000)
    my_sim.T.create_functions(my_sim.V_CG1)
    my_sim.create_H_fluxes()
    F = my_sim.F
    expressions = my_sim.expressions

    Kr_0 = expressions[0]
    E_Kr = expressions[1]
    Kr = Kr_0 * fenics.exp(-E_Kr/k_B/my_sim.T.T)
    test_sol = my_sim.v
    sol = my_sim.u
    expected_form = 0
    expected_form += -test_sol * (- Kr* sol**order)*my_sim.ds(1)
    expected_form += -test_sol*expressions[2]*my_sim.ds(1)
    expected_form += -test_sol*expressions[2]*my_sim.ds(2)
    assert expected_form.equals(F)
