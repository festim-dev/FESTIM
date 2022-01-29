import FESTIM
from FESTIM.post_processing import create_properties
import fenics


def test_create_properties():
    '''
    Test the function FESTIM.create_properties()
    '''
    mesh = fenics.UnitIntervalMesh(10)
    DG_1 = fenics.FunctionSpace(mesh, 'DG', 1)
    mat_1 = FESTIM.Material(1, D_0=1, E_D=0, S_0=7, E_S=0, thermal_cond=4, heat_capacity=5, rho=6, H={"free_enthalpy": 5, "entropy": 6})
    mat_2 = FESTIM.Material(2, D_0=2, E_D=0, S_0=8, E_S=0, thermal_cond=5, heat_capacity=6, rho=7, H={"free_enthalpy": 6, "entropy": 6})
    materials = FESTIM.Materials([mat_1, mat_2])
    mf = fenics.MeshFunction("size_t", mesh, 1, 0)
    for cell in fenics.cells(mesh):
        x = cell.midpoint().x()
        if x < 0.5:
            mf[cell] = 1
        else:
            mf[cell] = 2
    T = fenics.Expression("1", degree=1)
    D, thermal_cond, cp, rho, H, S = \
        create_properties(materials, mf, T)
    D = fenics.interpolate(D, DG_1)
    thermal_cond = fenics.interpolate(thermal_cond, DG_1)
    cp = fenics.interpolate(cp, DG_1)
    rho = fenics.interpolate(rho, DG_1)
    H = fenics.interpolate(H, DG_1)
    S = fenics.interpolate(S, DG_1)

    for cell in fenics.cells(mesh):
        assert D(cell.midpoint().x()) == mf[cell]
        assert thermal_cond(cell.midpoint().x()) == mf[cell] + 3
        assert cp(cell.midpoint().x()) == mf[cell] + 4
        assert rho(cell.midpoint().x()) == mf[cell] + 5
        assert H(cell.midpoint().x()) == mf[cell] + 10
        assert S(cell.midpoint().x()) == mf[cell] + 6
