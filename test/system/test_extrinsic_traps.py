import festim as F


def test_extrinsic_trap():
    """Runs a festim sim with an extrinsic trap"""
    my_materials = F.Materials([F.Material(id=1, D_0=2, E_D=1, name="mat")])
    my_mesh = F.MeshFromRefinements(10, 1)

    my_traps = F.ExtrinsicTrap(
        k_0=1,
        E_k=0.1,
        p_0=1e13,
        E_p=0.1,
        materials=["mat"],
        phi_0=2.5e19,
        n_amax=1e-1 * 6.3e28,
        f_a=1,
        eta_a=6e-4,
        n_bmax=1e-2 * 6.3e28,
        f_b=2,
        eta_b=2e-4,
    )

    my_temp = F.Temperature(300)

    my_settings = F.Settings(
        absolute_tolerance=1e10,
        relative_tolerance=1e-9,
        final_time=1,
    )
    my_dt = F.Stepsize(0.5)

    my_sim = F.Simulation(
        mesh=my_mesh,
        materials=my_materials,
        temperature=my_temp,
        settings=my_settings,
        traps=my_traps,
        dt=my_dt,
    )

    my_sim.initialise()
    my_sim.run()


def test_neutron_induced_trap():
    """Test to catch bug #434"""
    my_sim = F.Simulation()

    mesh = F.MeshFromVertices([0, 1, 2, 3, 4])
    my_sim.mesh = mesh

    my_sim.materials = F.Materials([F.Material(1, 1, 0, name="mat")])

    trap_1 = F.NeutronInducedTrap(
        0, 0, 0, 0, materials=["mat"], phi=F.x**2, K=1, n_max=10, A_0=0, E_A=0
    )
    my_sim.traps = F.Traps([trap_1])

    my_sim.boundary_conditions = []
    my_sim.T = F.Temperature(100)

    my_sim.settings = F.Settings(1e-10, 1e-10, final_time=10)
    my_sim.dt = F.Stepsize(1)

    # run simulation
    my_sim.initialise()
    my_sim.run()
