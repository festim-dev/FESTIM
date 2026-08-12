import numpy as np
import ufl

import festim as F

from .tools import error_L2, test_mesh_1d


def test_decay_reaction_MMS_transient():
    """Analytical verification of DecayReaction: a species undergoing radioactive
    decay follows c(x, t) = (1 + x) * exp(-lambda * t), with lambda = ln(2) /
    half_life. The linear spatial profile is represented exactly by P1 elements and
    is divergence-free (constant D), so no source term is needed and the remaining
    error is the first-order backward-Euler time discretisation."""
    # BUILD
    final_time = 0.1
    half_life = 0.1  # one half-life over the run: the concentration halves
    decay_constant = np.log(2) / half_life

    def u_exact(mod):
        return lambda x, t: (1 + x[0]) * mod.exp(-decay_constant * t)

    exact_ufl = u_exact(ufl)
    exact_np = lambda x: (1 + x[0]) * np.exp(-decay_constant * final_time)  # noqa: E731

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = test_mesh_1d
    mat = F.Material(name="mat", D_0=1, E_D=0)
    vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=mat)
    left = F.SurfaceSubdomain1D(id=1, x=0)
    right = F.SurfaceSubdomain1D(id=2, x=1)
    my_model.subdomains = [vol, left, right]

    H = F.Species("H")
    my_model.species = [H]
    my_model.temperature = 500  # constant, so D is constant and D * grad(c) is div-free

    my_model.boundary_conditions = [
        F.DirichletBC(subdomain=left, value=exact_ufl, species=H),
        F.DirichletBC(subdomain=right, value=exact_ufl, species=H),
    ]
    my_model.initial_conditions = [
        F.InitialConcentration(value=lambda x: 1 + x[0], species=H, volume=vol)
    ]
    my_model.reactions = [F.DecayReaction(reactant=H, half_life=half_life, volume=vol)]

    my_model.settings = F.Settings(atol=1e-12, rtol=1e-12, final_time=final_time)
    my_model.settings.stepsize = final_time / 100

    # RUN
    my_model.initialise()
    my_model.run()

    # TEST
    L2_error = error_L2(H.post_processing_solution, exact_np)
    assert L2_error < 3e-3
