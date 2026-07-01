from mpi4py import MPI

import numpy as np
import pytest
from dolfinx import fem, mesh

import festim as F

pyvista = pytest.importorskip("pyvista")
pyvista.OFF_SCREEN = True


def create_mock_solution():
    test_mesh = mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = fem.functionspace(test_mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.x.array[:] = np.ones_like(u.x.array)
    return u


def test_plot_single_species():
    species = F.Species("H")
    species.post_processing_solution = create_mock_solution()

    plotter = F.plot(species, show_edges=True, opacity=0.5)

    assert isinstance(plotter, pyvista.Plotter)
    assert plotter.shape == (1, 1)
    plotter.close()


def test_plot_multiple_species_creates_subplots():
    h = F.Species("H")
    d = F.Species("D")
    h.post_processing_solution = create_mock_solution()
    d.post_processing_solution = create_mock_solution()

    plotter = F.plot([h, d])

    assert plotter.shape == (1, 2)
    plotter.close()


def test_plot_subdomain_uses_subdomain_solution():
    material = F.Material(D_0=1, E_D=0)
    vol_1 = F.VolumeSubdomain(id=1, material=material)
    vol_2 = F.VolumeSubdomain(id=2, material=material)
    sol_1 = create_mock_solution()
    sol_2 = create_mock_solution()

    h = F.Species("H")
    h.subdomain_to_post_processing_solution = {vol_1: sol_1, vol_2: sol_2}

    plotter = F.plot(h, subdomain=vol_2)
    assert isinstance(plotter, pyvista.Plotter)
    plotter.close()


def test_plot_with_filename_saves_screenshot(tmp_path):
    species = F.Species("H")
    species.post_processing_solution = create_mock_solution()
    filename = tmp_path / "out.png"

    plotter = F.plot(species, filename=filename)
    plotter.close()
    assert filename.exists()


def test_plot_with_string_filename_saves_screenshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    species = F.Species("H")
    species.post_processing_solution = create_mock_solution()

    plotter = F.plot(species, filename="out.png")
    plotter.close()

    assert (tmp_path / "out.png").exists()


def test_plot_raises_for_invalid_field_type():
    with pytest.raises(
        TypeError,
        match=r"field must be of type festim\.Species or a list of festim\.Species",
    ):
        F.plot("H")


def test_plot_raises_if_no_solution():
    with pytest.raises(ValueError, match="has no post_processing_solution to plot"):
        F.plot(F.Species("H"))


def test_plot_default_show_edges_and_empty_name():
    species = F.Species()
    species.post_processing_solution = create_mock_solution()

    plotter = F.plot(species)

    assert isinstance(plotter, pyvista.Plotter)
    plotter.close()


def test_plot_default_with_several_subdomains():
    material = F.Material(D_0=1, E_D=0)
    vol_1 = F.VolumeSubdomain(id=1, material=material)
    vol_2 = F.VolumeSubdomain(id=2, material=material)
    sol_1 = create_mock_solution()
    sol_2 = create_mock_solution()

    h = F.Species("H")
    h.subdomain_to_post_processing_solution = {vol_1: sol_1, vol_2: sol_2}

    plotter = F.plot(h)
    assert isinstance(plotter, pyvista.Plotter)
    plotter.close()
