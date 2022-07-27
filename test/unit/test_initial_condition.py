import festim
import pytest


def test_error_initialisation_from_xdmf_missing_time_step():
    """
    Test that the function fails initialise_solutions if
    there's a missing key
    """

    with pytest.raises(ValueError, match=r"time_step"):
        festim.InitialCondition(value="my_file.xdmf", label="my_label", time_step=None)


def test_error_initialisation_from_xdmf_missing_label():
    """
    Test that the function fails initialise_solutions if
    there's a missing key
    """
    with pytest.raises(ValueError, match=r"label"):
        festim.InitialCondition(value="my_file.xdmf", label=None, time_step=1)
