import FESTIM
from fenics import *
import pytest
from pathlib import Path


def test_trap_density_xdmf_export_intergration_with_simultion(tmpdir):
    """Test"""
    density_expr = 2 + FESTIM.x**2 + 2 * FESTIM.y

    my_model = FESTIM.Simulation(log_level=20)
    my_model.mesh = FESTIM.Mesh()
    my_model.mesh.mesh = UnitSquareMesh(30, 30)
    mat_1 = FESTIM.Material(D_0=1, E_D=0, id=1)
    my_model.materials = FESTIM.Materials([mat_1])
    trap_1 = FESTIM.Trap(
        k_0=1, E_k=0, p_0=1, E_p=0, density=density_expr, materials=mat_1
    )
    my_model.traps = FESTIM.Traps([trap_1])
    my_model.T = FESTIM.Temperature(value=300)
    my_model.settings = FESTIM.Settings(
        transient=False,
        absolute_tolerance=1e06,
        relative_tolerance=1e-08,
    )
    density_file = tmpdir.join("density1.xdmf")
    my_export = FESTIM.TrapDensityXDMF(
        trap=trap_1,
        label="density1",
        filename=str(Path(density_file)),
    )
    my_exports = FESTIM.Exports([my_export])
    my_model.exports = my_exports
    my_model.initialise()
    my_model.run()

    V = FunctionSpace(my_model.mesh.mesh, "CG", 1)

    density_out = interpolate(FESTIM.as_expression(density_expr), V)

    density_in = Function(V)
    XDMFFile(str(Path(density_file))).read_checkpoint(density_in, "density1", -1)

    l2_error = errornorm(density_in, density_out, "L2")
    assert l2_error < 2e-3
