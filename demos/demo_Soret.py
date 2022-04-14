# Demo conditions from  https://doi.org/10.1063/5.0071935

from copyreg import constructor
import FESTIM as F
from fenics import ln, DOLFIN_EPS

my_model = F.Simulation()

#### MESH
my_model.mesh = F.MeshFromRefinements(
    initial_number_of_cells=6000,
    size=6e-3,
    refinements=[
        {
            "cells": 600,
            "x": 2e-6
        },
        {
            "cells": 100,
            "x": 20e-9
        }
    ]
)

#### PARAMETERS
import numpy as np

def thermal_cond(T):
    return 10.846*ln(T + DOLFIN_EPS)**2 - 184.22*ln(T + DOLFIN_EPS) + 872.47

constR=8.31446 # J/mol

def heat_transport(T):
    return -0.0065*constR*T*T

tungsten = F.Material(
    id=1,
    D_0=8.35e-8, #m2/s   
    E_D=0.06, #eV,
    heat_transport=heat_transport,
    thermal_cond=thermal_cond,
    borders=[0, 6e-3]
)
my_model.materials = F.Materials([tungsten])


#### THERMAL
my_model.T = F.HeatTransferProblem(transient=False)

heat_transfer_bcs = [
    F.DirichletBC(
        surfaces=[2],
        value=400.0, #K
        field="T"
    ),
    F.FluxBC(
        surfaces=[1],
        value=10e6, #W/m2
        field="T"
    )
]

#### DIFFUSIVE ATOM
x = F.x
flux = 1e21  # flux in He m-2 s-1
distribution = (x <= 10e-9) *  1e9/16.39 *  (7.00876507 + 0.6052078 * x*1e9 - 3.01711048 *(x*1e9)**2 + 1.36595786 * (x*1e9)**3 - 0.295595 * (x*1e9)**4 + 0.03597462 * (x*1e9)**5 - 0.0025142 * (x*1e9)**6 + 0.0000942235 * (x*1e9)**7 - 0.0000014679 * (x*1e9)**8)
source_term = distribution*flux
my_model.sources = [F.Source(value=source_term, field="solute", volume=1)]
my_model.boundary_conditions = heat_transfer_bcs + [F.DirichletBC(surfaces=[1, 2], value=0,field=0)]

#### SOLVER PARAMETERS
my_model.dt = F.Stepsize(
    initial_value=0.5,
    stepsize_change_ratio=1.1,
    dt_min=1e-05
)

my_model.settings = F.Settings(
    absolute_tolerance=1e10,
    relative_tolerance=1e-09,
    maximum_iterations=50,
    final_time=1e4,
    soret=True,
)

#### EXPORTS
folder = "test_Soret"

derived_quantities = F.DerivedQuantities(filename="{}/derived_quantities.csv".format(folder))
derived_quantities.derived_quantities = [
    F.TotalVolume(field="solute", volume=1),
    F.SurfaceFlux(field="solute", surface=1),
    F.SurfaceFlux(field="solute", surface=2),
]

txt_exports = F.TXTExports(
    fields=['solute', 'T'],
    labels=['solute', 'T'],
    times=[1e3,1e4],
    folder=folder
    )

my_model.exports = F.Exports([derived_quantities] + txt_exports.exports)

#### LET'S GO
my_model.initialise()
my_model.run()

