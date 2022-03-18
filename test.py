import FESTIM

my_model = FESTIM.Simulation(log_level=20)
my_model.mesh = FESTIM.MeshFromRefinements(10, size=1)
my_model.materials = FESTIM.Materials(
    [
        FESTIM.Material(id=1, D_0=1, E_D=0)
    ])
my_model.T = FESTIM.Temperature(100)
my_model.traps = FESTIM.Traps([
            FESTIM.CustomTrap(
                k_0=1,
                E_k=1,
                p_0=1,
                E_p=1,
                materials=1,
                form_parameters={
                    'prm1': 1,
                    'prm2': 2,
                })
                ])

my_model.boundary_conditions = [
    FESTIM.DirichletBC(surfaces=[1, 2], value=0)]
my_stepsize = FESTIM.Stepsize(1, stepsize_change_ratio=1.1, dt_min=1e-8)
my_model.dt = my_stepsize
my_model.settings = FESTIM.Settings(
    absolute_tolerance=1e-9,
    relative_tolerance=1e-9,
    final_time=1,
)
my_model.initialise()
my_model.run()
