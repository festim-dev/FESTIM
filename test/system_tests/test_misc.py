import dolfinx
import festim as F
import numpy as np


def test_petsc_options():
    my_model = F.HydrogenTransportProblem(
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "coucoucou": 3,
        }
    )

    tungsten = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)

    my_model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, num=100))
    vol1 = F.VolumeSubdomain1D(id=1, material=tungsten, borders=[0, 1])
    surface1 = F.SurfaceSubdomain1D(id=4, x=0)
    surface2 = F.SurfaceSubdomain1D(id=5, x=1)

    my_model.subdomains = [vol1, surface1, surface2]

    mobile = F.Species(name="H", mobile=True)

    my_model.species = [mobile]

    my_model.temperature = 600

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=surface1, species=mobile, value=1),
        F.FixedConcentrationBC(subdomain=surface2, species=mobile, value=0),
    ]

    my_model.settings = F.Settings(atol=1e-6, rtol=1e-6, transient=False)

    my_model.initialise()
    my_model.run()
