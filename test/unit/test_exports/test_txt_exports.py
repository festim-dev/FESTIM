from festim import TXTExports
import fenics as f
import os
import pytest
from pathlib import Path


class TestWrite:
    @pytest.fixture
    def my_export(self, tmpdir):
        d = tmpdir.mkdir("test_folder")
        my_export = TXTExports(
            ["solute", "T"], [1, 2, 3], ["solute_label", "T_label"], str(Path(d))
        )

        return my_export

    def test_txt_exports_times(self, my_export):
        for export in my_export.exports:
            assert export.times == my_export.times


def test_error_when_fields_and_labels_have_different_lengths():
    with pytest.raises(ValueError, match="Number of fields to be exported"):
        TXTExports(["solute", "T"], [1], ["solute_label"])
