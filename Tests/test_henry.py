from attr import has
from sympy.matrices import expressions
import FESTIM
import fenics
import pytest
import sympy as sp
import numpy as np


def test_henry_bc_varying_temperature():
	"""Checks the method DirichletBC.create_expression with type solubility Henry
	"""
	# build
	T = fenics.Constant(300)
	pressure_expr = 1e5*(1 + FESTIM.t)
	H_0_expr = 100
	E_H_expr = 0.5
	my_bc = FESTIM.HenrysBC(surface = 1, pressure = pressure_expr, H_0 = H_0_expr, E_H = E_H_expr)

	pressure_expr = fenics.Expression(sp.printing.ccode(pressure_expr),
										t = 0,
										degree = 1)
	H_0_expr = fenics.Expression(sp.printing.ccode(H_0_expr),
										t = 0,
										degree = 1)
	E_H_expr = fenics.Expression(sp.printing.ccode(E_H_expr),
										t = 0,
										degree = 1)

	# run
	my_bc.create_expression(T)
	# test

	def henrys(T, H_0, E_H, pressure):
		H = H_0*fenics.exp(-E_H/FESTIM.k_B/T)
		return H*pressure
	expected = FESTIM.BoundaryConditionExpression(T, eval_function=henrys, H_0=H_0_expr, E_H=E_H_expr, pressure=pressure_exp)
	assert my_bc.expression(0) == pytest.approx(expected(0))

	T.assign(1000)
	assert my_bc.expression(0) == pytest.approx(expected(0))
