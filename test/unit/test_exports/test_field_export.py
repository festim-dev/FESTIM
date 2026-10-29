"""Unit tests for the format-neutral field exports and their writer layer."""

from pathlib import Path

import pytest

import festim as F
from festim.exports import writers


@pytest.mark.parametrize(
    "format, backend, expected",
    [
        ("vtx", None, ".bp"),
        ("vtkhdf", None, ".vtkhdf"),
        ("xdmf", None, ".xdmf"),
        ("checkpoint", None, ".bp"),
        ("checkpoint", "adios2", ".bp"),
        ("checkpoint", "h5py", ".h5"),
    ],
)
def test_extension_follows_format(format, backend, expected):
    """The extension is corrected to match the chosen format."""
    with pytest.warns(UserWarning, match="does not have"):
        export = F.SpeciesExport(
            "my_export.wrong", field=F.Species("H"), format=format, backend=backend
        )
    assert export.filename == Path("my_export").with_suffix(expected)
    assert export.format == format


def test_correct_extension_does_not_warn(recwarn):
    """No warning when the filename already matches the format."""
    F.SpeciesExport("my_export.vtkhdf", field=F.Species("H"), format="vtkhdf")
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown format 'nope'"):
        F.SpeciesExport("my_export.bp", field=F.Species("H"), format="nope")


@pytest.mark.parametrize(
    "format, writer_class",
    [
        ("vtx", writers.VTXFieldWriter),
        ("vtkhdf", writers.VTKHDFFieldWriter),
        ("xdmf", writers.XDMFFieldWriter),
        ("checkpoint", writers.CheckpointFieldWriter),
    ],
)
def test_format_selects_writer(format, writer_class, tmp_path):
    """Each format maps to its writer class."""
    from festim.exports.field import _FORMAT_TO_WRITER

    assert _FORMAT_TO_WRITER[format] is writer_class


def test_checkpoint_backend_is_passed_to_writer(tmp_path):
    """The backend argument reaches the checkpoint writer."""
    export = F.SpeciesExport(
        str(tmp_path / "f.h5"),
        field=F.Species("H"),
        format="checkpoint",
        backend="h5py",
    )
    # define_writer builds the writer before touching the file
    writer = writers.CheckpointFieldWriter(export.filename, backend=export.backend)
    assert writer.backend == "h5py"


class TestBackwardsCompatibility:
    """The pre-existing class names must keep behaving exactly as before."""

    def test_vtx_species_export_is_vtx(self):
        with pytest.deprecated_call(match="VTXSpeciesExport is deprecated"):
            export = F.VTXSpeciesExport("my_export.bp", field=F.Species("H"))
        assert export.format == "vtx"
        assert export._checkpoint is False

    def test_vtx_temperature_export_is_vtx(self):
        with pytest.deprecated_call(match="VTXTemperatureExport is deprecated"):
            export = F.VTXTemperatureExport("T.bp")
        assert export.format == "vtx"

    def test_xdmf_export_is_xdmf(self):
        with pytest.deprecated_call(match="XDMFExport is deprecated"):
            export = F.XDMFExport("my_export.xdmf", field=F.Species("H"))
        assert export.format == "xdmf"

    def test_checkpoint_kwarg_maps_to_format(self):
        with pytest.deprecated_call(match="VTXSpeciesExport is deprecated"):
            export = F.VTXSpeciesExport(
                "my_export.bp", field=F.Species("H"), checkpoint=True
            )
        assert export.format == "checkpoint"
        assert export._checkpoint is True

    def test_custom_field_export_checkpoint_kwarg(self):
        with pytest.deprecated_call(match="format='checkpoint'"):
            export = F.CustomFieldExport(
                "my_export.bp", expression=lambda T: T, checkpoint=True
            )
        assert export.format == "checkpoint"
        # `checkpoint` used to be stored but ignored; now it reflects the format
        assert export.checkpoint is True


def test_subdomain_is_readable_on_every_field_export():
    """`subdomain` is used to name blocks, so every field export must expose it."""
    assert F.TemperatureExport("T.bp").subdomain is None
    assert F.SpeciesExport("H.bp", field=F.Species("H")).subdomain is None

    mat = F.Material(D_0=1, E_D=0)
    vol = F.VolumeSubdomain1D(id=3, borders=[0, 1], material=mat)
    export = F.SpeciesExport("H.bp", field=F.Species("H"), subdomain=vol)
    assert export.subdomain is vol
