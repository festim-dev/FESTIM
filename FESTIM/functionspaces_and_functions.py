from fenics import *


def create_function_spaces(mesh, nb_traps, element1='P', order1=1,
                           element2='P', degree2=1):
    ''' Returns FuncionSpaces for concentration and dynamic trap densities
    Arguments:
    - mesh: Mesh(), mesh of the functionspaces
    - nb_traps: int, number of traps
    - element1='P': string, the element of concentrations
    - order1=1: int, the order of the element of concentrations
    - element2='P': string, the element of dynamic trap densities
    - order1=2: int, the order of the element of dynamic trap densities
    '''
    if nb_traps == 0:
        V = FunctionSpace(mesh, element1, order1)
    else:
        V = VectorFunctionSpace(mesh, element1, order1, nb_traps + 1)
    W = FunctionSpace(mesh, element2, degree2)
    return V, W


def define_test_functions(V, W, number_ext_traps):
    '''
    Returns the testfunctions for formulation
    Arguments:
    - V, W: FunctionSpace(), functionspaces of concentrations and
    trap densities
    - number_int_traps: int, number of intrisic traps
    - number_ext_traps: int, number of extrinsic traps
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
    '''
    u = Function(V)
    # Split system functions to access components
    solutions = list(split(u))
    return u, solutions


def define_functions_extrinsic_traps(W, traps):
    '''
    Returns a list of Function(W)
    Arguments:
    -W: FunctionSpace, functionspace of trap densities
    -traps: dict, contains the traps infos
    '''
    extrinsic_traps = []

    for trap in traps:
        if 'type' in trap.keys():  # Default is intrinsic
            if trap['type'] == 'extrinsic':
                extrinsic_traps.append(Function(W))  # density
    return extrinsic_traps
