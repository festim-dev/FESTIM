import os

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import meshtags

import festim as F


def test_surface_flux_export_compute():
    """Test that the surface flux export computes the correct value"""

    # BUILD
    L = 4.0
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=4)

    # define mesh ds measure
    facet_indices = np.array(
        dummy_surface.locate_boundary_facet_indices(my_mesh.mesh),
        dtype=np.int32,
    ).flatten()
    tags_facets = np.array(
        [1],
        dtype=np.int32,
    ).flatten()
    facet_meshtags = meshtags(my_mesh.mesh, 0, facet_indices, tags_facets)
    ds = ufl.Measure("ds", domain=my_mesh.mesh, subdomain_data=facet_meshtags)

    # give function to species
    V = fem.functionspace(my_mesh.mesh, ("Lagrange", 1))
    c = fem.Function(V)
    c.interpolate(lambda x: 2 * x[0] ** 2 + 1)

    my_species = F.Species("H")
    my_species.solution = c

    my_export = F.SurfaceFlux(
        field=my_species,
        surface=dummy_surface,
    )
    my_export.D = D

    # RUN
    my_export.compute(my_species.solution, ds=ds)

    # TEST
    # flux = -D grad(c)_ \cdot n = -D dc/dx = -D * 4 * x
    expected_value = -D * 4 * dummy_surface.x
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)


@pytest.mark.parametrize("value", ["my_export.csv", "my_export.txt"])
def test_title_generation(tmp_path, value):
    """Test that the title is made to be written to the header in a csv or txt file"""
    my_export = F.SurfaceFlux(
        filename=os.path.join(tmp_path, f"{value}"),
        field=F.Species("TEST"),
        surface=F.SurfaceSubdomain1D(id=35, x=1),
    )
    my_export.value = 2.0
    my_export.write(0)
    title = np.genfromtxt(my_export.filename, delimiter=",", max_rows=1, dtype=str)

    expected_title = "TEST flux surface 35"

    assert title[1] == expected_title


def test_write_overwrite(tmp_path):
    """Test that the write method overwrites the file if it already exists"""
    filename = os.path.join(tmp_path, "my_export.csv")
    my_export = F.SurfaceFlux(
        filename=filename,
        field=F.Species("test"),
        surface=F.SurfaceSubdomain1D(id=1, x=0),
    )
    my_export.value = 2.0
    my_export.write(0)
    my_export.write(1)

    my_export2 = F.SurfaceFlux(
        filename=filename,
        field=F.Species("test"),
        surface=F.SurfaceSubdomain1D(id=1, x=0),
    )
    my_export2.value = 3.0
    my_export2.write(1)
    my_export2.write(2)
    my_export2.write(3)

    data = np.genfromtxt(filename, delimiter=",", names=True)
    file_length = data.size
    expected_length = 3

    assert file_length == expected_length


def test_filename_setter_raises_TypeError():
    """Test that a TypeError is raised when the filename is not a string"""

    with pytest.raises(TypeError, match="filename must be of type str"):
        F.SurfaceQuantity(
            filename=1,
            field=F.Species("test"),
            surface=F.SurfaceSubdomain1D(id=1, x=0),
        )


def test_filename_setter_raises_ValueError(tmp_path):
    """Test that a ValueError is raised when the filename does not end with .csv or .txt"""

    with pytest.raises(ValueError):
        F.SurfaceQuantity(
            filename=os.path.join(tmp_path, "my_export.xdmf"),
            field=F.Species("test"),
            surface=F.SurfaceSubdomain1D(id=1, x=0),
        )


def test_field_setter_raises_TypeError():
    """Test that a TypeError is raised when the field is not a F.Species"""

    with pytest.raises(TypeError):
        F.SurfaceQuantity(
            field=1,
            surface=F.SurfaceSubdomain1D(id=1, x=0),
        )


@pytest.mark.parametrize("value", ["my_export.csv", "my_export.txt"])
def test_writer(tmp_path, value):
    """Test that the writes values at each timestep to either a csv or txt file"""
    my_export = F.SurfaceFlux(
        filename=os.path.join(tmp_path, f"{value}"),
        field=F.Species("test"),
        surface=F.SurfaceSubdomain1D(id=1, x=0),
    )
    my_export.value = 2.0

    for i in range(10):
        my_export.write(i)
        file_length = len(np.genfromtxt(my_export.filename, delimiter=","))

        expected_length = i + 2

        assert file_length == expected_length


def test_surface_setter_raises_TypeError():
    """Test that a TypeError is raised when the surface is not a
    F.SurfaceSubdomain"""

    with pytest.raises(
        TypeError, match="surface should be an int or F.SurfaceSubdomain"
    ):
        F.SurfaceQuantity(
            field=F.Species("H"),
            surface="1",
        )
