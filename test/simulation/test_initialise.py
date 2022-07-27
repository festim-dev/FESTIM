import festim as F


def test_initialise_changes_nb_of_sources():
    """Creates a Simulation object with a HeatTransferProblem that has sources,
    calls initialise several times and checks that the number of heat sources
    is correct.
    Reprodces bug in issue #473
    """
    my_model = F.Simulation()
    my_model.mesh = F.MeshFromVertices([1, 2, 3])
    my_model.materials = F.Materials([F.Material(id=1, D_0=1, E_D=0, thermal_cond=1)])
    my_model.T = F.HeatTransferProblem(transient=False)
    # add source to the HeatTransferProblem
    my_model.sources = [F.Source(value=0, volume=1, field="T")]
    my_model.settings = F.Settings(
        transient=False,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
    )

    for _ in range(4):
        my_model.initialise()
        assert len(my_model.T.sources) == 1


def test_initialise_sets_t_to_zero():
    """Creates a Simulation object and checks that .initialise() sets
    the t attribute to zero
    """
    # build
    my_model = F.Simulation()
    my_model.mesh = F.MeshFromVertices([1, 2, 3])
    my_model.materials = F.Materials([F.Material(id=1, D_0=1, E_D=0, thermal_cond=1)])
    my_model.T = F.Temperature(100)
    my_model.settings = F.Settings(
        transient=False,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
    )

    # assign a value to my_model.t
    my_model.t = 2

    # run
    my_model.initialise()

    # test

    # check that my_model.t is reinitialised to zero
    assert my_model.t == 0


def test_initialise_initialise_dt():
    """Creates a Simulation object and checks that .initialise() sets
    the value attribute of the dt attribute to dt.initial_value
    """
    # build
    my_model = F.Simulation()
    my_model.mesh = F.MeshFromVertices([1, 2, 3])
    my_model.materials = F.Material(id=1, D_0=1, E_D=0, thermal_cond=1)
    my_model.T = F.Temperature(100)
    my_model.dt = F.Stepsize(initial_value=3)
    my_model.settings = F.Settings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, final_time=4
    )
    my_model.dt.value.assign(26)

    # run
    my_model.initialise()

    # test
    assert my_model.dt.value(2) == 3
