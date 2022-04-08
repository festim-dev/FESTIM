import FESTIM as F

sim = F.Simulation()

sim.mesh = F.MeshFromVertices([0, 1, 2, 3, 4])

sim.materials = F.Materials([F.Material(1, 1, 1)])

sim.exports = F.Exports(
    [
        F.XDMFExport("solute", label="mobile"),
        F.XDMFExport("T", label="temperature", filename="temperature_file.xdmf"),
    ]
)

sim.T = F.Temperature(100)

sim.settings = F.Settings(1e10, 1e-10, transient=False)

sim.initialise()

sim.run()
