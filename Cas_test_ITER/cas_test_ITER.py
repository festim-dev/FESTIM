from context import FESTIM
from fenics import *
import sympy as sp
from parameters import parameters


if __name__ == "__main__":
    FESTIM.generic_simulation.run(parameters, log_level=30)
