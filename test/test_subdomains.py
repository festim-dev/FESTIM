from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest

import festim as F


def test_different_surface_ids():
    """Checks that different surface ids are correctly set"""
    my_test_model = F.HydrogenTransportProblem()
    my_test_model.species = [F.Species("H")]
    L = 1e-04
    my_test_model.mesh = F.Mesh1D(np.linspace(0, L, num=3))

    surface_subdomains_ids = [3, 8]
    surface_subdomain_1 = F.SurfaceSubdomain1D(id=surface_subdomains_ids[0], x=0)
    surface_subdomain_2 = F.SurfaceSubdomain1D(id=surface_subdomains_ids[1], x=L)
    volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=None)
    my_test_model.subdomains = [
        surface_subdomain_1,
        surface_subdomain_2,
        volume_subdomain,
    ]

    my_test_model.define_function_spaces()
    my_test_model.define_meshtags_and_measures()

    for surf_id in surface_subdomains_ids:
        assert surf_id in np.array(my_test_model.facet_meshtags.values)


def test_different_volume_ids():
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 10, 11))

    vol_subdom_ids = [2, 16, 7]
    vol_subdomain_1 = F.VolumeSubdomain1D(
        id=vol_subdom_ids[0], borders=[0, 2], material=None
    )
    vol_subdomain_2 = F.VolumeSubdomain1D(
        id=vol_subdom_ids[1], borders=[2, 6], material=None
    )
    vol_subdomain_3 = F.VolumeSubdomain1D(
        id=vol_subdom_ids[2], borders=[6, 10], material=None
    )
    my_model.subdomains = [vol_subdomain_1, vol_subdomain_2, vol_subdomain_3]

    my_model.define_meshtags_and_measures()

    for vol_id in vol_subdom_ids:
        assert vol_id in np.array(my_model.volume_meshtags.values)


def test_non_matching_volume_borders():
    """Checks that non-matching borders raise an error"""
    mesh = F.Mesh1D(vertices=np.linspace(0, 5, 6))
    vol_subdomain_1 = F.VolumeSubdomain1D(id=1, borders=[0, 2], material=None)
    vol_subdomain_2 = F.VolumeSubdomain1D(id=1, borders=[3, 5], material=None)
    subdomains = [vol_subdomain_1, vol_subdomain_2]

    with pytest.raises(ValueError, match="Subdomain borders don't match to each other"):
        mesh.check_borders(subdomains)


def test_matching_volume_borders_non_ascending_order():
    """Checks that subdomain placed in non ascending order still passes"""
    mesh = F.Mesh1D(vertices=np.linspace(0, 8, 9))
    vol_subdomain_1 = F.VolumeSubdomain1D(id=1, borders=[0, 2], material=None)
    vol_subdomain_2 = F.VolumeSubdomain1D(id=2, borders=[4, 8], material=None)
    vol_subdomain_3 = F.VolumeSubdomain1D(id=3, borders=[2, 4], material=None)
    subdomains = [vol_subdomain_1, vol_subdomain_2, vol_subdomain_3]

    mesh.check_borders(subdomains)


def test_borders_out_of_domain():
    """Checks that borders outside of the domain raise an error"""
    mesh = F.Mesh1D(vertices=np.linspace(0, 2))
    subdomains = [F.VolumeSubdomain1D(id=1, borders=[1, 15], material=None)]
    with pytest.raises(ValueError, match="borders dont match domain borders"):
        mesh.check_borders(subdomains)


def test_borders_inside_domain():
    """Checks that borders inside of the domain raise an error"""
    mesh = F.Mesh1D(vertices=np.linspace(0, 20))
    subdomains = [F.VolumeSubdomain1D(id=1, borders=[1, 6], material=None)]
    with pytest.raises(ValueError, match="borders dont match domain borders"):
        mesh.check_borders(subdomains)


def test_raise_error_with_no_volume_subdomain():
    """Checks that error is rasied when no volume subdomain is defined"""
    mesh = F.Mesh1D(vertices=np.linspace(0, 20))

    with pytest.raises(ValueError, match="No volume subdomains defined"):
        mesh.check_borders([])


@pytest.mark.parametrize("input", [1, 2, 3, 4])
def test_find_volume_from_int(input):
    """test that the correct volume is returned when input is an int"""

    vol_1 = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=None)
    vol_2 = F.VolumeSubdomain1D(id=2, borders=[1, 2], material=None)
    vol_3 = F.VolumeSubdomain1D(id=3, borders=[2, 3], material=None)
    vol_4 = F.VolumeSubdomain1D(id=4, borders=[3, 4], material=None)

    volumes = [vol_1, vol_2, vol_3, vol_4]

    assert F.find_volume_from_id(input, volumes) == volumes[input - 1]


def test_ValueError_raised_when_id_not_found_in_volumes_subdomains():
    """test that a ValueError is raised when an id is not found in the list of volume subdomains"""

    volumes = [F.VolumeSubdomain1D(id=1, borders=[0, 1], material=None)]

    with pytest.raises(ValueError, match="id 5 not found in list of volumes"):
        F.find_volume_from_id(5, volumes)


def test_ValueError_rasied_when_volume_ids_are_not_unique():
    """Checks"""
    my_test_model = F.HydrogenTransportProblem(
        mesh=F.Mesh1D(np.linspace(0, 2, num=10)), species=[F.Species("H")]
    )

    vol_1 = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=None)
    vol_2 = F.VolumeSubdomain1D(id=1, borders=[1, 2], material=None)
    my_test_model.subdomains = [
        vol_1,
        vol_2,
    ]

    my_test_model.define_function_spaces()

    with pytest.raises(ValueError, match="Volume ids are not unique"):
        my_test_model.define_meshtags_and_measures()


def test_ValueError_raised_when_id_not_found_in_surface_subdomains():
    """test that a ValueError is raised when an id is not found in the list of surface subdomains"""

    surfaces = [F.SurfaceSubdomain(id=1)]

    with pytest.raises(ValueError, match="id 3 not found in list of surfaces"):
        F.find_surface_from_id(3, surfaces)


@pytest.mark.parametrize("input_id, output_index", [(1, 2), (4, 0), (7, 1), (9, 3)])
def test_find_surface_from_id(input_id, output_index):
    """test that the correct surface is returned when input is an int"""

    surf_1 = F.SurfaceSubdomain(id=7)
    surf_2 = F.SurfaceSubdomain1D(id=4, x=0)
    surf_3 = F.SurfaceSubdomain1D(id=9, x=1)
    surf_4 = F.SurfaceSubdomain(id=1)

    surfaces = [surf_2, surf_1, surf_4, surf_3]

    assert F.find_surface_from_id(input_id, surfaces) == surfaces[output_index]


def test_volume_subdomain_properties():
    """Tests that the volume subdomain property obtains the correct
    subdomains from the the model subdomains list"""

    my_model = F.HydrogenTransportProblem()
    my_model.subdomains = [
        F.SurfaceSubdomain(id=7),
        F.SurfaceSubdomain1D(id=4, x=0),
        F.SurfaceSubdomain(id=2),
        F.VolumeSubdomain(id=1, material=None),
        F.VolumeSubdomain1D(id=9, borders=[0, 1], material=None),
    ]

    assert len(my_model.volume_subdomains) == 2
    for subdomain in my_model.volume_subdomains:
        assert isinstance(subdomain, F.VolumeSubdomain)

    assert len(my_model.surface_subdomains) == 3
    for subdomain in my_model.surface_subdomains:
        assert isinstance(subdomain, F.SurfaceSubdomain)


mesh1d = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, nx=1)
mesh2d = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx=1, ny=1)
mesh3d = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, nx=1, ny=1, nz=1)
