from FESTIM import HenrysBC, t
import fenics as f
import sympy as sp


surfaces=[1,2]
Test_henry = HenrysBC(surfaces,0,1,1e1)
print(Test_henry.pressure)
print(sp.printing.ccode(Test_henry.pressure))
