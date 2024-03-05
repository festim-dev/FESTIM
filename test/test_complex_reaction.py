import numpy as np
import festim as F


def concentration_A_exact(t, c_A_0, k, p):
    """Analytical solution for the concentration of species A in a reaction A + B <-> C + D
    assuming [A]_0 = [B]_0 and [C]_0 = [D]_0 = 0

    Args:
        t (float or ndarray): time
        c_A_0 (float): initial concentration of A
        k (float): forward reaction rate
        p (float): backward reaction rate

    Returns:
        ndarray: the concentration of A at a given time
    """

    A = 1 - p / k
    B = 2 * p / k * c_A_0
    C = -p / k * c_A_0**2
    roots = np.roots([A, B, C])

    if len(roots) == 1:
        a = roots[0]
        b = roots[0]
    else:
        a, b = roots[0], roots[1]
    correction_factor = (
        1 - p / k
    )  # TODO verify why this.... but checks out with scipy.solve_ivp
    F = ((c_A_0 - a) / (c_A_0 - b)) * np.exp(-(a - b) * correction_factor * k * t)
    return (a - b * F) / (1 - F)


def model_test_reaction(stepsize):
    """Creates a festim model with a single reaction and runs it.
    The reaction is A + B <-> C + D

    Args:
        stepsize (float): the stepsize

    Returns:
        festim.HydrogenTransportProblem: the model
    """

    my_model = F.HydrogenTransportProblem()

    # -------- Mesh --------- #

    L = 1
    vertices = np.linspace(0, L, num=10)
    my_model.mesh = F.Mesh1D(vertices)

    # -------- Materials and subdomains --------- #

    mat = F.Material(D_0=1, E_D=0, name="mat")

    my_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, L], material=mat)
    left_surface = F.SurfaceSubdomain1D(id=1, x=0)
    right_surface = F.SurfaceSubdomain1D(id=2, x=L)

    my_model.subdomains = [
        my_subdomain,
        left_surface,
        right_surface,
    ]

    # -------- Hydrogen species and reactions --------- #

    species_A = F.Species("A")
    species_B = F.Species("B")
    species_C = F.Species("C")
    species_D = F.Species("D")

    my_model.species = [
        species_A,
        species_B,
        species_C,
        species_D,
    ]

    my_model.reactions = [
        F.Reaction(
            k_0=350e-4,
            E_k=0,
            p_0=120e-4,
            E_p=0,
            reactant1=species_A,
            reactant2=species_B,
            product=[species_C, species_D],
            volume=my_subdomain,
        ),
    ]

    # -------- Temperature --------- #

    my_model.temperature = 400

    # -------- Boundary conditions --------- #

    my_model.boundary_conditions = []

    # -------- Initial conditions --------- #

    my_model.initial_conditions = [
        F.InitialCondition(value=1, species=species_A),
        F.InitialCondition(value=1, species=species_B),
    ]

    # -------- Exports --------- #

    total_A = F.TotalVolume(species_A, my_subdomain)
    total_B = F.TotalVolume(species_B, my_subdomain)
    total_C = F.TotalVolume(species_C, my_subdomain)
    total_D = F.TotalVolume(species_D, my_subdomain)

    my_model.exports = [total_A, total_B, total_C, total_D]

    # -------- Settings --------- #

    my_model.settings = F.Settings(
        atol=1e-10, rtol=1e-10, max_iterations=30, final_time=200
    )

    my_model.settings.stepsize = F.Stepsize(initial_value=stepsize)

    # -------- Run --------- #

    my_model.initialise()

    my_model.run()

    return my_model


def compute_error(model):
    """Computes the relative L2 error norm between the analytical solution and the numerical solution

    Args:
        model (F.HydrogenTransportModel): the numerical festim model

    Returns:
        float: the relative L2 error norm
    """
    L = model.mesh.vertices[-1]
    k = model.reactions[0].k_0
    p = model.reactions[0].p_0
    c_A_0 = model.initial_conditions[0].value
    total_A = model.exports[0]
    times = np.array(total_A.t)
    data_total_A = total_A.data
    data_total_A = np.array(data_total_A)
    concentration_A_analytical = concentration_A_exact(t=times, c_A_0=c_A_0, k=k, p=p)

    # relative l2 error norm
    error = np.sqrt(
        np.sum((concentration_A_analytical - data_total_A / L) ** 2)
    ) / np.sqrt(np.sum(concentration_A_analytical**2))

    return error


def test_reaction():
    stepsize = 1
    model = model_test_reaction(stepsize)
    error = compute_error(model)
    assert error < 1e-2
