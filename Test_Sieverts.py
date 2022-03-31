from FESTIM import SievertsBC, t
import fenics as f
import sympy as sp


surfaces = [1,2]
sievert_test = SievertsBC(surfaces,1,1,1)
sievert_test.pressure = 1e2* (t <= 1.0)
print(sievert_test.pressure)
print(sp.printing.ccode(sievert_test.pressure))
