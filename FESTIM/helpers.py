import FESTIM
import warnings
import fenics as f
import xml.etree.ElementTree as ET
warnings.simplefilter('always', DeprecationWarning)


def update_expressions(expressions, t):
    '''Update all FEniCS Expression() in expressions.

    Arguments:
    - expressions: list, contains the fenics Expression
    to be updated.
    - t: float, time.
    '''
    for expression in expressions:
        expression.t = t
    return expressions


bc_types = {
    "dc": ["dc", "solubility", "dc_imp", "dc_custom"],
    "neumann": ["flux", "flux_custom"],
    "robin": ["recomb", "flux_custom"]
}

T_bc_types = {
    "dc": ["dc", "dc_custom"],
    "neumann": ["flux", "flux_custom"],
    "robin": ["flux_custom", "convective_flux"]
}

quantity_types = [
    "surface_flux",
    "average_volume",
    "average_surface",
    "maximum_volume",
    "minimum_volume",
    "total_volume",
    "total_surface"
    ]

field_types = [
    "solute",
    "retention",
    "T"
]
parameters_helper = {
    "materials": {
        "H": {
            "description": "[insert description]",
            "unit": "[insert unit]"
        },
        "E_D": {
            "description": "Diffusion coefficient activation energy",
            "unit": "eV"
            },
        "D_0": {
            "description": "Diffusion coefficient pre exponential factor",
            "unit": "(m^2/s)"
            },
        "S_0": {
            "description": "Solubility coefficient pre exponential factor",
            "unit": "(m^2/s)"
            },
        "E_S": {
            "description": "Solubility coefficient activation energy",
            "unit": "eV"
            },
        "borders": {
            "description": "1D only: delimitations of the domain. \
            Exemple: [0, 0.5]",
            "unit": "m"
            },
        "thermal_cond": {
            "description": "required if heat equation is solved or if thermal\
                 flux is computed. thermal conductivity",
            "unit": "W/m/K"
            },
        "heat_capacity": {
            "description": "required if heat equation is solved. \
                Heat capacity",
            "unit": "J/K/kg"
            },
        "rho": {
            "description": "required if heat equation is solved. \
                Density",
            "unit": "kg/m^3"
            },
        "id": {
            "description": "id of the domain. If a marked mesh is given,\
                id must correspond to the volume_markers_file",
            "unit": "None"
            },
    },
    "traps": {
        "E_k": {
            "description": "Trapping rate activation energy",
            "unit": "eV"
            },
        "k_0": {
            "description": "Trapping rate pre-exponential factor",
            "unit": "m^3/s"
            },
        "E_p": {
            "description": "Detrapping rate activation energy",
            "unit": "eV"
            },
        "p_0": {
            "description": "Detrapping rate pre-exponential factor",
            "unit": "s^-1"
            },
        "density": {
            "description": "denisty of the trap. Can be float or an expression\
                 (ex: (1 + FESTIM.x)*FESTIM.t<100)",
            "unit": "m^-3"
            },
        "materials": {
            "description": "ids of the domains where the trap is present.\
                 Can be int or list of int",
            "unit": "None"
            },
        "source_term": {
            "description": "Volumetric source term for the trapped population",
            "unit": "m^-3.s^-1"
            },
        },
    "boundary_conditions": {
        "type": {
            "dc": "Dirichlet boundary condition",
            "flux": "Pure Neumann boundary condition",
            "recomb": "Recombination flux",
            "convective_flux": "Convective exchange (for heat transfer)",
            "solubility": "Dirichlet boundary condition based on solubility and pressure c=S*P^0.5",
            "table": "Dirichlet boundary condition based on interpolated values from 2D table (t, c(t)).",
            "dc_imp": "Dirichlet boundary condition based on triangular model for volumetric implantation"
            },
        "surfaces": {
            "description": "List of surfaces on which the boundary condition is applied",
            "unit": "None"
            },
        "value": {
                    "description": "Value of boundary conditions only needed for types dc and flux. Can be float or an expression (ex: (1 + FESTIM.x)*FESTIM.t<100)",
                    "unit": "m^-3 or m^-2.s^-1"
                },
        "component": {
                    "description": "int in [0;N], N being the number of traps. By default 0, the solute population",
                    "unit": "None"
                },
        "Kr_0": {
                    "description": "Value of recombination coefficient pre-exponential factor",
                    "unit": "m^(-2+3*n).s^-1 , where n is the order of recombination"
                },
        "E_Kr": {
                    "description": "Value of recombination coefficient activation energy",
                    "unit": "eV"
                },
        "S_0": {
                    "description": "Value of solubility coefficient pre-exponential factor",
                    "unit": "m^-3.Pa^-0.5"
                },
        "E_S": {
                    "description": "Value of solubility coefficient activation energy",
                    "unit": "eV"
                },
        "pressure": {
                    "description": "Value of pressure",
                    "unit": "Pa"
        }
    },
    "mesh_parameters": {
        "refinements": {},
        "size": {
            "description": "Size of the 1D problem",
            "unit": "m"
        },
        "initial_number_of_cells": {
            "description": "Number of cells in the domain before refinement",
            "unit": "cells"
        },
        "mesh_file": {},
        "cells_file": {},
        "facets_file": {},
        "meshfunction_cells": {},
        "meshfunction_facets": {},
    },
    "temperature": {},
    "solving_parameters": {},
    "exports": {},
    }


def help_key(key):
    def find_dict(key, var, res=[]):
        if key in var:
            res.append(var[key])
        else:
            for k in var.keys():
                if type(var[k]) == dict:
                    find_dict(key, var=var[k], res=res)
        return res
    d = find_dict(key, parameters_helper)
    for k in d:
        print(k)


def read_parameters(simulation, parameters):
    msg = "The use of parameters will soon be deprecated \
            please use the object-oriented approach instead"
    warnings.warn(msg, DeprecationWarning)
    create_settings(simulation, parameters)
    create_stepsize(simulation, parameters)
    create_concentration_objects(simulation, parameters)
    create_boundarycondition_objects(simulation, parameters)
    create_materials(simulation, parameters)
    create_temperature(simulation, parameters)
    create_initial_conditions(simulation, parameters)
    define_mesh(simulation, parameters)
    create_exports(simulation, parameters)
    create_sources_objects(simulation, parameters)


def create_stepsize(self, parameters):
    """Creates FESTIM.Stepsize object from a parameters dict.
    To be deprecated.

    Args:
        parameters (dict): parameters dict (<= 0.7.1)
    """
    if self.settings.transient:
        self.dt = FESTIM.Stepsize()
        if "solving_parameters" in parameters:
            solving_parameters = parameters["solving_parameters"]
            self.dt.value.assign(solving_parameters["initial_stepsize"])
            if "adaptive_stepsize" in solving_parameters:
                self.dt.adaptive_stepsize = {}
                for key, val in solving_parameters["adaptive_stepsize"].items():
                    self.dt.adaptive_stepsize[key] = val
                if "t_stop" not in solving_parameters["adaptive_stepsize"]:
                    self.dt.adaptive_stepsize["t_stop"] = None
                if "stepsize_stop_max" not in solving_parameters["adaptive_stepsize"]:
                    self.dt.adaptive_stepsize["stepsize_stop_max"] = None


def create_settings(self, parameters):
    """Creates FESTIM.Settings object from a parameters dict.
    To be deprecated.

    Args:
        parameters (dict): parameters dict (<= 0.7.1)
    """
    my_settings = FESTIM.Settings(None, None)
    if "solving_parameters" in parameters:
        # Check if transient
        solving_parameters = parameters["solving_parameters"]
        if "type" in solving_parameters:
            if solving_parameters["type"] == "solve_transient":
                my_settings.transient = True
            elif solving_parameters["type"] == "solve_stationary":
                my_settings.transient = False
                self.dt = None
            else:
                raise ValueError(
                    str(solving_parameters["type"]) + ' unkown')

        # Declaration of variables
        if my_settings.transient:
            my_settings.final_time = solving_parameters["final_time"]

        my_settings.absolute_tolerance = solving_parameters["newton_solver"]["absolute_tolerance"]
        my_settings.relative_tolerance = solving_parameters["newton_solver"]["relative_tolerance"]
        my_settings.maximum_iterations = solving_parameters["newton_solver"]["maximum_iterations"]
        if "traps_element_type" in solving_parameters:
            my_settings.traps_element_type = solving_parameters["traps_element_type"]

        if "update_jacobian" in solving_parameters:
            my_settings.update_jacobian = solving_parameters["update_jacobian"]

        if "soret" in parameters["temperature"]:
            my_settings.soret = parameters["temperature"]["soret"]
        if "materials" in parameters:
            if "S_0" in parameters["materials"][0]:
                my_settings.chemical_pot = True
    self.settings = my_settings


def create_concentration_objects(self, parameters):
    """Creates FESTIM.Mobile and FESTIM.Traps objects from a parameters
    dict. self.mobile is created in Simulation.init
    To be deprecated.

    Args:
        parameters (dict): parameters dict (<= 0.7.1)
    """
    traps = []
    if "traps" in parameters:
        for trap in parameters["traps"]:
            if "type" in trap:
                traps.append(FESTIM.ExtrinsicTrap(**{key: val for key,
                                                  val in trap.items() if
                                                  key != "type"}))
            else:
                traps.append(
                    FESTIM.Trap(trap["k_0"], trap["E_k"], trap["p_0"], trap["E_p"], trap["materials"], trap["density"])
                )
    self.traps = FESTIM.Traps(traps)


def create_sources_objects(self, parameters):
    """Creates a list of FESTIM.Source objects from a parameters
    dict.
    To be deprecated.

    Args:
        parameters (dict): parameters dict (<= 0.7.1)
    """
    self.sources = []
    if "traps" in parameters:
        for i, trap in enumerate(parameters["traps"], 1):
            if "source_term" in trap:
                if type(trap["materials"]) is not list:
                    materials = [trap["materials"]]
                else:
                    materials = trap["materials"]
                for mat in materials:
                    self.sources.append(
                        FESTIM.Source(trap["source_term"], mat, i)
                    )
    if "source_term" in parameters:
        if isinstance(parameters["source_term"], dict):
            for mat in self.materials.materials:
                if type(mat.id) is not list:
                    vols = [mat.id]
                else:
                    vols = mat.id
                for vol in vols:
                    self.sources.append(
                        FESTIM.Source(parameters["source_term"]["value"], volume=vol, field="0")
                    )
        elif isinstance(parameters["source_term"], list):
            for source_dict in parameters["source_term"]:
                if type(source_dict["volume"]) is not list:
                    vols = [source_dict["volume"]]
                else:
                    vols = source_dict["volume"]
                for volume in vols:
                    self.sources.append(
                        FESTIM.Source(source_dict["value"], volume=volume, field="0")
                    )
    if "temperature" in parameters:
        if "source_term" in parameters["temperature"]:
            for source in parameters["temperature"]["source_term"]:
                self.sources.append(
                    FESTIM.Source(source["value"], source["volume"], "T")
                )


def create_materials(self, parameters):
    """Creates a FESTIM.Materials object from a parameters
    dict.
    To be deprecated.

    Args:
        parameters (dict): parameters dict (<= 0.7.1)
    """
    materials = []
    if "materials" in parameters:
        for material in parameters["materials"]:
            my_mat = FESTIM.Material(**material)
            materials.append(my_mat)
    self.materials = FESTIM.Materials(materials)
    derived_quantities = {}
    if "exports" in parameters:
        if "derived_quantities" in parameters["exports"]:
            derived_quantities = parameters["exports"]["derived_quantities"]
    temp_type = "expression"  # default temperature type is expression
    if "temperature" in parameters:
        if "type" in parameters["temperature"]:
            temp_type = parameters["temperature"]["type"]
    self.materials.check_materials(temp_type, derived_quantities)


def create_boundarycondition_objects(self, parameters):
    """Creates a list of FESTIM.BoundaryCondition objects from a
    parameters dict.
    To be deprecated.

    Args:
        parameters (dict): parameters dict (<= 0.7.1)
    """
    self.boundary_conditions = []
    if "boundary_conditions" in parameters:
        for BC in parameters["boundary_conditions"]:
            bc_type = BC["type"]
            if bc_type in FESTIM.helpers.bc_types["dc"]:
                if bc_type == "solubility":
                    my_BC = FESTIM.SievertsBC(
                        **{key: val for key, val in BC.items()
                            if key != "type"})
                elif bc_type == "dc_imp":
                    my_BC = FESTIM.ImplantationDirichlet(
                        **{key: val for key, val in BC.items()
                            if key != "type"})
                else:
                    my_BC = FESTIM.DirichletBC(
                        **{key: val for key, val in BC.items()
                            if key != "type"})
            elif bc_type not in FESTIM.helpers.bc_types["neumann"] or \
                    bc_type not in FESTIM.helpers.bc_types["robin"]:
                if bc_type == "recomb":
                    my_BC = FESTIM.RecombinationFlux(
                        **{key: val for key, val in BC.items()
                            if key != "type"}
                    )
                else:
                    my_BC = FESTIM.FluxBC(
                        **{key: val for key, val in BC.items()
                            if key != "type"}
                    )
            self.boundary_conditions.append(my_BC)

    if "temperature" in parameters:
        if "boundary_conditions" in parameters["temperature"]:

            BCs = parameters["temperature"]["boundary_conditions"]
            for BC in BCs:
                bc_type = BC["type"]
                if bc_type in FESTIM.helpers.T_bc_types["dc"]:
                    my_BC = FESTIM.DirichletBC(
                        component="T",
                        **{key: val for key, val in BC.items()
                            if key != "type"})
                elif bc_type not in FESTIM.helpers.T_bc_types["neumann"] or \
                        bc_type not in FESTIM.helpers.T_bc_types["robin"]:
                    if bc_type == "convective_flux":
                        my_BC = FESTIM.ConvectiveFlux(
                            **{key: val for key, val in BC.items()
                                if key != "type"})
                    else:
                        my_BC = FESTIM.FluxBC(
                            component="T",
                            **{key: val for key, val in BC.items()
                                if key != "type"})
                self.boundary_conditions.append(my_BC)


def create_temperature(self, parameters):
    """Creates a FESTIM.Temperature object from a
    parameters dict.
    To be deprecated.

    Args:
        parameters (dict): parameters dict (<= 0.7.1)
    """
    if "temperature" in parameters:
        temp_type = parameters["temperature"]["type"]
        if temp_type == "expression":
            self.T = FESTIM.Temperature(parameters["temperature"]['value'])
            # self.T.expression = parameters["temperature"]['value']
        else:
            self.T = FESTIM.HeatTransferProblem()
            self.T.bcs = [bc for bc in self.boundary_conditions if bc.component == "T"]
            if temp_type == "solve_transient":
                self.T.transient = True
                self.T.initial_value = parameters["temperature"]["initial_condition"]
            elif temp_type == "solve_stationary":
                self.T.transient = False
            if "source_term" in parameters["temperature"]:
                self.T.source_term = parameters["temperature"]["source_term"]


def create_initial_conditions(self, parameters):
    """Creates a list of FESTIM.InitialCondition objects from a
    parameters dict.
    To be deprecated.

    Args:
        parameters (dict): parameters dict (<= 0.7.1)
    """
    initial_conditions = []
    if "initial_conditions" in parameters.keys():
        for condition in parameters["initial_conditions"]:
            initial_conditions.append(FESTIM.InitialCondition(**condition))
    self.initial_conditions = initial_conditions


def create_exports(self, parameters):
    """Creates a FESTIM.Exports object from a
    parameters dict.
    To be deprecated.

    Args:
        parameters (dict): parameters dict (<= 0.7.1)
    """
    self.exports = FESTIM.Exports([])
    if "exports" in parameters:
        if "xdmf" in parameters["exports"]:
            export = parameters["exports"]["xdmf"]
            mode = 1
            if "last_timestep_only" in export:
                mode = "last"
            if "nb_iterations_between_exports" in export:
                mode = export["nb_iterations_between_exports"]
            my_xdmf_exports = FESTIM.XDMFExports(
                **{key: val for key, val in
                    export.items()
                    if key not in [
                        "last_timestep_only",
                        "nb_iterations_between_exports"
                        ]
                   },
                mode=mode
                )
            self.exports.exports += my_xdmf_exports.xdmf_exports

        if "derived_quantities" in parameters["exports"]:
            derived_quantities = FESTIM.DerivedQuantities(**parameters["exports"]["derived_quantities"])
            self.exports.exports.append(derived_quantities)

        if "txt" in parameters["exports"]:
            txt_exports = FESTIM.TXTExports(**parameters["exports"]["txt"])
            self.exports.exports += txt_exports.exports

        if "error" in parameters["exports"]:
            for error_dict in parameters["exports"]["error"]:
                for field, exact in zip(error_dict["fields"], error_dict["exact_solutions"]):
                    error = FESTIM.Error(field, exact, error_dict["norm"], error_dict["degree"])
                    self.exports.exports.append(error)


def define_mesh(self, parameters):
    """Creates a FESTIM.Mesh object from a
    parameters dict.
    To be deprecated.

    Args:
        parameters (dict): parameters dict (<= 0.7.1)
    """
    if "mesh_parameters" in parameters:
        mesh_parameters = parameters["mesh_parameters"]

        if "volume_file" in mesh_parameters.keys():
            self.mesh = FESTIM.MeshFromXDMF(**mesh_parameters)
        elif ("mesh" in mesh_parameters.keys() and
                isinstance(mesh_parameters["mesh"], type(f.Mesh()))):
            self.mesh = FESTIM.Mesh(**mesh_parameters)
        elif "vertices" in mesh_parameters.keys():
            self.mesh = FESTIM.MeshFromVertices(mesh_parameters["vertices"])
        else:
            self.mesh = FESTIM.MeshFromRefinements(**mesh_parameters)


def kJmol_to_eV(energy):
    """Converts an energy value given in units kJ mol^{-1} to eV

    Args:
        energy (float): Energy in kJ mol^{-1}

    Returns:
        energy (float): Energy in eV
    """
    energy_in_eV = FESTIM.k_B*energy*1e3/FESTIM.R

    return energy_in_eV


def extract_xdmf_times(filename):
    """Returns a list of timesteps in an XDMF file

    Args:
        filename (str): the XDMF filename (must end with .xdmf)

    Returns:
        list: the timesteps
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    domains = list(root)
    domain = domains[0]
    grids = list(domain)
    grid = grids[0]

    times = []
    for c in grid:
        for element in c:
            if "Time" in element.tag:
                times.append(float(element.attrib["Value"]))
    return times


def extract_xdmf_labels(filename):
    """Returns a list of labels in an XDMF file

    Args:
        filename (str): the XDMF filename (must end with .xdmf)

    Returns:
        list: the labels
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    domains = list(root)
    domain = domains[0]
    grids = list(domain)
    grid = grids[0]

    labels = []
    for c in grid:
        for element in c:
            if "Attribute" in element.tag:
                labels.append(element.attrib["Name"])

    unique_labels = list(set(labels))
    return unique_labels
