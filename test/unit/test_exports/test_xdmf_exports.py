from festim import XDMFExports
import pytest


def test_error_different_lengths_functions_labels():
    with pytest.raises(
        ValueError, match="doesn't match number of labels in xdmf exports"
    ):
        XDMFExports(["solute", "T"], ["solute"], "my_folder")


class TestListXDMFExports:
    labels = ["T", "solute", "1"]
    functions = ["T", "solute", 1]
    folder = "my_folder"
    my_exports = XDMFExports(functions, labels, folder=folder)

    def test_length(self):
        assert len(self.my_exports.xdmf_exports) == len(self.functions)

    def test_function_attributes(self):
        for export, function in zip(self.my_exports.xdmf_exports, self.functions):
            assert export.field == function

    def test_label_attributes(self):
        for export, label in zip(self.my_exports.xdmf_exports, self.labels):
            assert export.label == label

    def test_folder_attributes(self):
        for export in self.my_exports.xdmf_exports:
            assert export.folder == self.folder


# def test_deprecation_warning():
#     with pytest.warns(DeprecationWarning) as record:
#         XDMFExports(functions=["coucou"], labels=["coucou"], folder="my_folder")
#         assert len(record) == 1
