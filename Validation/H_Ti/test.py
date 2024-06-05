import festim as F
import fenics as f
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

################### PARAMETERS ###################
N_A_const = 6.022e23  # Avogadro, mol^-1
e = 1.602e-19
M_H2 = 2.016e-3 / N_A_const  # the H2 mass, kg mol^-1

# Sample size
A = 1e-2 * 1.3e-2  # Ti surface area (1cm x 1.3cm), m^2
L = 1e-3  # Ti thickness, m
V = A * L  # Ti volume (1cm x 1.3cm x 1mm), m^-3

# Ti properties

N_b = 9.4e4 * N_A_const  #  the number of atomic sites per unit of volume of Ti, m^-3
N_s = (
    2.16e-5 * N_A_const
)  # the number of atomic sites per unit of surface area of Ti, m^-2
n_Ti = N_b * V  #  the number of moles of Ti
lambda_Ti = N_s / N_b

# Properties of fluxes
E_diff = F.kJmol_to_eV(5.3e4 / 1e3)  # diffusion activation energy, eV
D0 = 9e-7  # Diffusion pre-factor for D in W, m^2 s^-1

E_des = F.kJmol_to_eV(1.17e5 / 1e3)  # activateion energy for desotpion, eV
k_des = 2.16e8 * N_A_const  # frequency factor for the surface desorption

E_sb = F.kJmol_to_eV(
    1.36e5 / 1e3
)  # the activation energy value for the inward subsurface transport, eV
k_sb = 8.56e12  # frequency factor for the inward subsurface transport, s^-1

E_bs = F.kJmol_to_eV(
    1.6e5 / 1e3
)  # the activation energy value for the outward subsurface transport, eV
k_bs = 7.77e13  # frequency factor for the outward subsurface transport, s^-1

# Chamber
V_ch = 2.95e-3  # the chamber volume, m^-3
P0 = 1.3e4  # the initial pressure, Pa

################### FUNCTIONS ###################


def S0(T):
    # the capturing coefficient
    return 0.0143 * f.exp(F.kJmol_to_eV(1.99) / F.k_B / T)


def P_H2(T, X):
    # partial pressure of hydrogen, Pa
    X0 = 0
    return (
        F.k_B * T * e / V_ch * (P0 * V_ch / (F.k_B * T * e) + (X0 - X) * (2 * n_Ti) / 2)
    )


def J_vs(surf_conc, T, X):
    J_ads = (
        2
        * S0(T)
        * (1 - surf_conc / N_s) ** 2
        * P_H2(T, X)
        / (2 * np.pi * M_H2 * F.k_B * T * e) ** 0.5
    )
    J_des = k_des * (surf_conc / N_s) ** 2 * f.exp(-E_des / F.k_B / T)
    return J_ads - J_des


def J_vs1(surf_conc, T, X):
    J_ads = (
        2
        * S0(T)
        * (1 - surf_conc / N_s) ** 2
        * P_H2(T, X)
        / (2 * np.pi * M_H2 * F.k_B * T * e) ** 0.5
    )
    J_des = k_des * (surf_conc / N_s) ** 2 * f.exp(-E_des / F.k_B / T)
    return J_ads / 100 - J_des


################### CUSTOM MODEL CLASS ###################
class CustomSimulation(F.Simulation):
    def iterate(self):
        super().iterate()

        # Content
        X = (f.assemble(self.mobile.solution * self.mesh.dx)) * A / n_Ti

        # Update surface and subsurface concentrations
        self.h_transport_problem.surface_concentrations[0].prms["X"].assign(X)
        self.h_transport_problem.surface_concentrations[1].prms["X"].assign(X)


################### MODEL ###################
for i, T0 in enumerate([450 + 273, 500 + 273, 550 + 273, 600 + 273, 650 + 273]):
    Ti_model_impl = CustomSimulation(log_level=40)

    # Mesh
    vertices = np.linspace(0, L, num=251)
    Ti_model_impl.mesh = F.MeshFromVertices(np.sort(vertices))

    # Materials
    Ti_model_impl.materials = F.Material(id=1, D_0=D0, E_D=E_diff)

    surf_conc = F.SurfaceConcentration(
        k_sb=k_sb,
        E_sb=E_sb,
        k_bs=k_bs,
        E_bs=E_bs,
        l_abs=lambda_Ti,
        N_s=N_s,
        N_b=N_b,
        J_vs=J_vs,
        surfaces=1,
        initial_condition=F.InitialCondition(field="adsorbed", value=0),
        X=0,
    )

    surf_conc1 = F.SurfaceConcentration(
        k_sb=k_sb,
        E_sb=E_sb,
        k_bs=k_bs,
        E_bs=E_bs,
        l_abs=lambda_Ti,
        N_s=N_s,
        N_b=N_b,
        J_vs=J_vs1,
        surfaces=2,
        initial_condition=F.InitialCondition(field="adsorbed", value=0),
        X=0,
    )

    Ti_model_impl.surface_concentrations = [surf_conc, surf_conc1]

    Ti_model_impl.T = F.Temperature(value=T0)

    Ti_model_impl.dt = F.Stepsize(
        initial_value=1e-9, stepsize_change_ratio=1.1, max_stepsize=5, dt_min=1e-4
    )

    Ti_model_impl.settings = F.Settings(
        absolute_tolerance=1e11,
        relative_tolerance=1e-5,
        maximum_iterations=50,
        final_time=25 * 60,
    )

    # Exports
    n_exp = 10
    results_folder = "test/"

    derived_quantities = [
        F.DerivedQuantities(
            [
                F.TotalVolume(field="solute", volume=1),
                F.AdsorbedHydrogen(surface=1),
                F.AdsorbedHydrogen(surface=2),
            ],
            nb_iterations_between_compute=1,
            filename=results_folder + f"derived_quantities_expl_{T0}.csv",
        )
    ]

    Ti_model_impl.exports = derived_quantities

    Ti_model_impl.initialise()
    Ti_model_impl.run()

    data_impl = pd.read_csv(f"./test/derived_quantities_expl_{T0}.csv", header=0)
    for i in range(1, 3):
        plt.plot(
            data_impl["t(s)"] / 60,
            data_impl[f"Concentration of adsorbed H on surface {i}"] / N_s,
        )

    plt.xscale("log")
    plt.show()

    f.plot(Ti_model_impl.h_transport_problem.mobile.solution)
    plt.show()
