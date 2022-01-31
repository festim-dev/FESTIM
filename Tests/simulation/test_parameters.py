import FESTIM
import pytest


def error_unknown_solving_type():
    with pytest.raises(ValueError, match="unknwon"):
        FESTIM.Simulation({"solving_parameters": {"type": "coucou"}})
