import FESTIM as F
import numpy as np


def test_convective_flux():
    sim = F.Simulation()

    sim.mesh = F.MeshFromVertices(np.linspace(0, 1, num=50))

    sim.T = F.HeatTransferProblem(transient=False)

    T_external = 2
    sim.boundary_conditions = [
        F.ConvectiveFlux(h_coeff=10, T_ext=T_external, surfaces=1),
        F.DirichletBC(surfaces=2, value=T_external + 1, field="T"),
    ]

    sim.materials = F.Materials([F.Material(1, D_0=1, E_D=0, thermal_cond=2)])

    sim.exports = F.Exports([F.XDMFExport("T", checkpoint=False)])

    sim.settings = F.Settings(1e-10, 1e-10, transient=False)

    sim.initialise()
    sim.run()

    assert sim.T.T(0) > T_external
