import festim as F
import dolfinx
import ufl
import numpy as np


def test_assign_functions_to_species():
    """Test that checks if the function assign_functions_to_species
    creates the correct attributes for each species
    """


    mesh = F.Mesh1D(vertices=np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]))
    model = F.HydrogenTransportProblem(
        mesh=mesh,
        species=[
            F.Species(name="H"),
            # F.Species(name="Trap"),
            ],
    )
    model.define_function_space()
    model.assign_functions_to_species()

    for spe in model.species:
        assert spe.solution is not None
        assert spe.prev_solution is not None
        assert spe.test_function is not None
        assert isinstance(spe.solution, dolfinx.fem.Function)
        assert isinstance(spe.prev_solution, dolfinx.fem.Function)
        assert isinstance(spe.test_function, ufl.Argument)