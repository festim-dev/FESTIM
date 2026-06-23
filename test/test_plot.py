import importlib
import sys
import types

import pytest

import festim as F


class FakePlotter:
    def __init__(self, shape=None):
        self.shape = shape
        self.mesh_calls = []
        self.subplot_calls = []
        self.text_calls = []
        self.show_called = False
        self.screenshot_filename = None

    def subplot(self, row, col):
        self.subplot_calls.append((row, col))

    def add_mesh(self, grid, **kwargs):
        self.mesh_calls.append((grid, kwargs))

    def view_xy(self):
        pass

    def add_text(self, text, font_size):
        self.text_calls.append((text, font_size))

    def show(self):
        self.show_called = True

    def screenshot(self, filename):
        self.screenshot_filename = filename


def _setup_fake_pyvista(monkeypatch, off_screen=False):
    fake_pyvista = types.SimpleNamespace(
        OFF_SCREEN=off_screen,
        Plotter=FakePlotter,
        UnstructuredGrid=object,
    )
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)
    return fake_pyvista


def test_plot_single_species(monkeypatch):
    _setup_fake_pyvista(monkeypatch, off_screen=False)
    plot_module = importlib.import_module("festim.plot")
    monkeypatch.setattr(plot_module, "_make_ugrid", lambda solution, pyvista_module: 1)

    species = F.Species("H")
    species.post_processing_solution = object()
    plotter = F.plot(species, show_edges=True)

    assert plotter.shape is None
    assert len(plotter.mesh_calls) == 1
    assert plotter.mesh_calls[0][1]["show_edges"] is True
    assert plotter.show_called is True


def test_plot_multiple_species_creates_subplots(monkeypatch):
    _setup_fake_pyvista(monkeypatch, off_screen=False)
    plot_module = importlib.import_module("festim.plot")
    monkeypatch.setattr(plot_module, "_make_ugrid", lambda solution, pyvista_module: 1)

    h = F.Species("H")
    d = F.Species("D")
    h.post_processing_solution = object()
    d.post_processing_solution = object()
    plotter = F.plot([h, d])

    assert plotter.shape == (1, 2)
    assert plotter.subplot_calls == [(0, 0), (0, 1)]
    assert len(plotter.mesh_calls) == 2


def test_plot_subdomain_uses_subdomain_solution(monkeypatch):
    _setup_fake_pyvista(monkeypatch, off_screen=False)
    plot_module = importlib.import_module("festim.plot")

    used_solutions = []

    def fake_make_ugrid(solution, pyvista_module):
        used_solutions.append(solution)
        return 1

    monkeypatch.setattr(plot_module, "_make_ugrid", fake_make_ugrid)

    material = F.Material(D_0=1, E_D=0)
    vol_1 = F.VolumeSubdomain(id=1, material=material)
    vol_2 = F.VolumeSubdomain(id=2, material=material)
    sol_1 = object()
    sol_2 = object()
    h = F.Species("H")
    h.subdomain_to_post_processing_solution = {vol_1: sol_1, vol_2: sol_2}

    F.plot(h, subdomain=vol_2)
    assert used_solutions == [sol_2]


def test_plot_with_filename_saves_screenshot(monkeypatch, tmp_path):
    _setup_fake_pyvista(monkeypatch, off_screen=True)
    plot_module = importlib.import_module("festim.plot")
    monkeypatch.setattr(plot_module, "_make_ugrid", lambda solution, pyvista_module: 1)

    species = F.Species("H")
    species.post_processing_solution = object()
    filename = tmp_path / "out.png"
    plotter = F.plot(species, filename=filename)

    assert plotter.screenshot_filename == str(filename)
    assert plotter.show_called is False


def test_plot_raises_for_invalid_field_type(monkeypatch):
    _setup_fake_pyvista(monkeypatch, off_screen=False)
    with pytest.raises(TypeError):
        F.plot("H")


def test_plot_raises_if_no_solution(monkeypatch):
    _setup_fake_pyvista(monkeypatch, off_screen=False)
    with pytest.raises(ValueError):
        F.plot(F.Species("H"))
