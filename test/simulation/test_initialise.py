import festim as F
from pathlib import Path
import pytest


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


def test_TXTExport_times_added_to_milestones(tmpdir):
    """Creates a Simulation object and checks that, if no dt.milestones
     are given and TXTExport.times are given, TXTExport.times are
    are added to dt.milestones by .initialise()
    """
    # tmpdir
    d = tmpdir.mkdir("test_folder")

    # build
    my_model = F.Simulation()
    my_model.mesh = F.MeshFromVertices([1, 2, 3])
    my_model.materials = F.Material(id=1, D_0=1, E_D=0, thermal_cond=1)
    my_model.T = F.Temperature(100)
    my_model.dt = F.Stepsize(initial_value=3)
    my_model.settings = F.Settings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, final_time=4
    )
    txt_export = F.TXTExport(
        field="solute",
        filename="{}/solute_label.txt".format(str(Path(d))),
        times=[1, 2, 3],
    )
    my_model.exports = [txt_export]

    # run
    my_model.initialise()

    # test
    assert my_model.dt.milestones == txt_export.times


@pytest.mark.parametrize(
    "quantity",
    [
        F.SurfaceFlux(field="solute", surface=1),
        F.TotalVolume(field="solute", volume=1),
        F.TotalSurface(field="solute", surface=1),
        F.AverageSurface(field="solute", surface=1),
        F.AverageVolume(field="solute", volume=1),
        F.HydrogenFlux(surface=1),
        F.ThermalFlux(surface=1),
    ],
)
@pytest.mark.parametrize("sys", ["cylindrical", "spherical"])
def test_cartesian_and_surface_flux_warning(quantity, sys):
    """Creates a Simulation object and checks that, if either a cylindrical
    or spherical meshes are given with a SurfaceFlux, a warning is raised.
    """
    # build
    my_model = F.Simulation()
    my_model.mesh = F.MeshFromVertices([1, 2, 3], type=sys)
    my_model.materials = F.Material(id=1, D_0=1, E_D=0)
    my_model.T = F.Temperature(100)
    my_model.dt = F.Stepsize(initial_value=3)
    my_model.settings = F.Settings(
        absolute_tolerance=1e-10, relative_tolerance=1e-10, final_time=4
    )

    derived_quantities = F.DerivedQuantities([quantity])
    my_model.exports = [derived_quantities]

    # test
    with pytest.warns(UserWarning, match="Some derived quantities .* non-cartesian"):
        my_model.initialise()
