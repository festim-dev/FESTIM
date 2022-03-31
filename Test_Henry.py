from FESTIM import HenrysBC, k_B
import fenics as f
import sympy as sp


def henrys_law(T, H_0, E_H, pressure):
	H = H_0*f.exp(-E_H/k_B/T)
	return H*pressure


surfaces=[1,2]
E_S = 0.5
H_0 = 1e-4
Temp = 300
T = f.Constant(Temp)
pressure = 1e4
Test_henry = HenrysBC(surfaces,H_0,E_S,pressure)

Test_henry.create_expression(T)

print(Test_henry.pressure)
print(sp.printing.ccode(Test_henry.pressure))
print(Test_henry.expression(0)-henrys_law(Temp,H_0,E_S,pressure))
