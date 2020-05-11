from fenics import *


def create_function_space(mesh, nb_traps, element_solute='CG', order_solute=1,
                          element_trap='CG', order_trap=1):
    ''' Returns FuncionSpaces for concentration and dynamic trap densities
    Arguments:
    - mesh: fenics.Mesh(), mesh of the functionspaces
    - nb_traps: int, number of traps
    - element_solute='CG': string, the element of solute concentration
    - order_solute=1: int, the order of the element of solute concentration
    - element_trap='CG': string, the element of traps concentrations
    - order_trap=1: int, the order of the element of traps concentrations
    Returns:
    - V: fenics.FunctionSpace(), the function space of concentrations
    '''
    if nb_traps == 0:
        V = FunctionSpace(mesh, element_solute, order_solute)
    else:
        solute = FiniteElement(element_solute, mesh.ufl_cell(), order_solute)
        traps = FiniteElement(element_trap, mesh.ufl_cell(), order_trap)
        element = [solute] + [traps]*nb_traps
        V = FunctionSpace(mesh, MixedElement(element))
    return V


def define_test_functions(V, W, number_ext_traps):
    '''
    Returns the testfunctions for formulation
    Arguments:
    - V, W: FunctionSpace(), functionspaces of concentrations and
    trap densities
    - number_ext_traps: int, number of extrinsic traps
    Returns:
    - testfunctions_concentrations: list, contains fenics.TestFunction()
    - testfunctions_extrinsic_traps: list, contains fenics.TestFunction()
    '''
    v = TestFunction(V)
    testfunctions_concentrations = list(split(v))
    testfunctions_extrinsic_traps = list()
    for i in range(number_ext_traps):
        testfunctions_extrinsic_traps.append(TestFunction(W))
    return testfunctions_concentrations, testfunctions_extrinsic_traps


def define_functions(V):
    '''
    Returns Function() objects for formulation
    Arguments:
    - V: fenics.FunctionSpace()
    Returns:
    - u: fenics.Function()
    - solutions: list, components of u
    '''
    u = Function(V)
    # Split system functions to access components
    solutions = list(split(u))
    return u, solutions


def define_functions_extrinsic_traps(W, traps):
    '''
    Returns a list of Function(W)
    Arguments:
    -W: fenics.FunctionSpace, functionspace of trap densities
    -traps: dict, contains the traps infos
    Returns:
    - extrinsic_traps: list, contains fenics.Function()
    '''
    extrinsic_traps = []

    for trap in traps:
        if 'type' in trap.keys():  # Default is intrinsic
            if trap['type'] == 'extrinsic':
                extrinsic_traps.append(Function(W))  # density
    return extrinsic_traps
