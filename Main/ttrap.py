from fenics import *
from dolfin import *
import numpy as np
import sympy as sp
import csv
import sys
import os
import argparse


class Ttrap():

    def save_as(self):
        '''
        - parameters : none
        - returns filedescription : string of the saving path
        '''
        valid = False
        while valid is False:
            print("Save as (.csv):")
            filedesorption = input()
            if filedesorption == '':
                filedesorption = "desorption.csv"
            if filedesorption.endswith('.csv'):
                valid = True
                try:
                    with open(filedesorption, 'r') as f:
                        print('This file already exists.'
                              ' Do you want to replace it ? (y/n)')
                    choice = input()
                    if choice == "n" or choice == "N":
                        valid = False
                    elif choice != "y" and choice != "Y":
                        valid = False
                except:
                    valid = True
            else:
                print("Please enter a file ending with the extension .csv")
                valid = False
        return filedesorption

    def export_TDS(self, filedesorption, desorption):
        '''
        - filedesorption : string, the path of the csv file.
        - desorption : list, values to be exported.
        '''
        busy = True
        while busy is True:
            try:
                with open(filedesorption, "w+") as output:
                    busy = False
                    writer = csv.writer(output, lineterminator='\n')
                    writer.writerows(['dTt'])
                    for val in desorption:
                        writer.writerows([val])
            except:
                print("The file " + filedesorption +
                      " might currently be busy."
                      "Please close the application then press any key")
                input()
        return

    def export_txt(self, filename, function):
        '''
        Exports a 1D function into a txt file.
        Arguments:
        - filemame : str
        - function : FEniCS Function
        Returns:
        - True on sucess,
        - False on failure
        '''
        export = Function(W)
        export = project(function)
        busy = True
        while busy is True:
            try:
                np.savetxt(filename + '.txt', np.transpose(
                            [x.vector()[:], export.vector()[:]]))
                return True
            except:
                print("The file " + filename + ".txt might currently be busy."
                      "Please close the application then press any key.")
                input()

        return False

    def export_profiles(self, res, exports, t, dt):
        '''
        Exports 1D profiles in txt files.
        Arguments:
        - res: list, contains FEniCS Functions
        - exports: dict, defined by define_exports()
        - t: float, time
        - dt: FEniCS Constant(), stepsize
        Returns:
        - dt: FEniCS Constant(), stepsize
        '''
        functions = exports['txt']['functions']
        labels = exports['txt']['labels']
        if len(functions) != len(labels):
            raise NameError("Number of functions to be exported "
                            "doesn't match number of labels in txt exports")

        [_u_1, _u_2, _u_3, _u_4, _u_5, _u_6, retention] = res

        solution_dict = {
            'solute': _u_1,
            '1': _u_2,
            '2': _u_3,
            '3': _u_4,
            '4': _u_5,
            '5': _u_6,
            'retention': retention
        }
        times = sorted(exports['txt']['times'])
        end = True
        for time in times:
            if t == time:
                if times.index(time) != len(times)-1:
                    next_time = times[times.index(time)+1]
                    end = False
                else:
                    end = True
                for i in range(len(functions)):
                    solution = solution_dict[functions[i]]
                    label = labels[i]
                    self.export_txt(
                        label + '_' + str(t) + 's', solution)
                break
            if t < time:
                next_time = time
                end = False
                break
        if end is False:
            if t + float(dt) > next_time:
                dt.assign(time - t)
        return dt

    def define_xdmf_files(self, exports, folder):
        '''
        Returns a list of XDMFFile
        Arguments:
        - exports: dict, defined by define_exports()
        - folder: str, defined by define_exports()
        '''
        if len(exports['xdmf']['functions']) != len(exports['xdmf']['labels']):
            raise NameError("Number of functions to be exported "
                            "doesn't match number of labels in xdmf exports")
        files = list()
        for i in range(0, len(exports["xdmf"]["functions"])):
            u_file = XDMFFile(folder + exports["xdmf"]["labels"][i] + '.xdmf')
            u_file.parameters["flush_output"] = True
            u_file.parameters["rewrite_function_mesh"] = False
            files.append(u_file)
        return files

    def export_xdmf(self, res, exports, files, t):
        '''
        Exports the solutions fields in xdmf files.
        Arguments:
        - res: list, contains FEniCS Functions
        - exports: dict, defined by define_exports()
        - files: list, contains XDMFFile
        - t: float

        '''
        [_u_1, _u_2, _u_3, _u_4, _u_5, _u_6, retention] = res

        solution_dict = {
            'solute': _u_1,
            '1': _u_2,
            '2': _u_3,
            '3': _u_4,
            '4': _u_5,
            '5': _u_6,
            'retention': retention
        }

        for i in range(0, len(exports["xdmf"]["functions"])):
            label = exports["xdmf"]["labels"][i]
            function = exports["xdmf"]["functions"][i]
            solution = solution_dict[exports["xdmf"]["functions"][i]]
            solution.rename(label, "label")
            files[i].write(solution, t)
        return

    def find_material_from_id(self, materials, id):
        ''' Returns the material from a given id
        Parameters:
        - materials : list of dicts
        - id : int
        '''
        for material in materials:
            if material['id'] == id:
                return material
                break
        print("Couldn't find ID " + str(id) + " in materials list")
        return

    def formulation(self, traps, solutions, testfunctions, previous_solutions):
        ''' Creates formulation for trapping MRE model.
        Parameters:
        - traps : dict, contains the energy, density and domains
        of the traps
        - solutions : list, contains the solution fields
        - testfunctions : list, contains the testfunctions
        - previous_solutions : list, contains the previous solution fields

        Returns:
        - F : variational formulation
        '''
        F = 0
        F += ((u_1 - u_n1) / dt)*v_1*dx
        for material in materials:
            D_0 = material['D_0']
            E_diff = material['E_diff']
            subdomain = material['id']
            F += D_0 * exp(-E_diff/k_B/temp) * \
                dot(grad(u_1), grad(v_1))*dx(subdomain)
        F += - flux_*f*v_1*dx

        i = 1
        for trap in traps:
            trap_density = trap['density']
            energy = trap['energy']
            material = trap['materials']
            F += ((solutions[i] - previous_solutions[i]) / dt) * \
                testfunctions[i]*dx
            if type(material) is list:
                for subdomain in material:
                    corresponding_material = \
                        self.find_material_from_id(materials, subdomain)
                    D_0 = corresponding_material['D_0']
                    E_diff = corresponding_material['E_diff']
                    alpha = corresponding_material['alpha']
                    beta = corresponding_material['beta']
                    F += - D_0 * exp(-E_diff/k_B/temp)/alpha/alpha/beta*u_1 * \
                        (trap_density - solutions[i]) * \
                        testfunctions[i]*dx(subdomain)
                    F += v_0*exp(-energy/k_B/temp)*solutions[i] * \
                        testfunctions[i]*dx(subdomain)
            else:
                subdomain = trap['materials']
                corresponding_material = \
                    self.find_material_from_id(materials, subdomain)
                D_0 = corresponding_material['D_0']
                E_diff = corresponding_material['E_diff']
                alpha = corresponding_material['alpha']
                beta = corresponding_material['beta']
                F += - D_0 * exp(-E_diff/k_B/temp)/alpha/alpha/beta*u_1 * \
                    (trap_density - solutions[i]) * \
                    testfunctions[i]*dx(subdomain)
                F += v_0*exp(-energy/k_B/temp)*solutions[i] * \
                    testfunctions[i]*dx(subdomain)
            F += ((solutions[i] - previous_solutions[i]) / dt)*v_1*dx
            i += 1
        return F

    def subdomains(self, mesh, materials):
        '''
        Iterates through the mesh and mark them
        based on their position in the domain
        Arguments:
        - mesh : the mesh
        - materials : list, contains the dictionaries of the materials
        Returns :
        - volume_markers : MeshFunction that contains the subdomains
            (0 if no domain was found)
        - measurement_dx : the measurement dx based on volume_markers
        - surface_markers : MeshFunction that contains the surfaces
        - measurement_ds : the measurement ds based on surface_markers
        '''
        volume_markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
        for cell in cells(mesh):
            for material in materials:
                if cell.midpoint().x() >= material['borders'][0] \
                 and cell.midpoint().x() <= material['borders'][1]:
                    volume_markers[cell] = material['id']

        measurement_dx = dx(subdomain_data=volume_markers)

        surface_markers = MeshFunction(
            "size_t", mesh, mesh.topology().dim()-1, 0)
        surface_markers.set_all(0)
        i = 0
        for f in facets(mesh):
            i += 1
            x0 = f.midpoint()
            surface_markers[f] = 0
            if near(x0.x(), 0):
                surface_markers[f] = 1
            if near(x0.x(), size):
                surface_markers[f] = 2
        measurement_ds = ds(subdomain_data=surface_markers)
        return volume_markers, measurement_dx, surface_markers, measurement_ds

    def mesh_and_refine(self, mesh_parameters):
        '''
        Mesh and refine iteratively until meeting the refinement
        conditions.
        Arguments:
        - mesh_parameters : dict, contains initial number of cells, size,
        and refinements (number of cells and position)
        Returns:
        - mesh : the refined mesh.
        '''
        print('Meshing ...')
        initial_number_of_cells = mesh_parameters["initial_number_of_cells"]
        size = mesh_parameters["size"]
        mesh = IntervalMesh(initial_number_of_cells, 0, size)
        if "refinements" in mesh_parameters:
            for refinement in mesh_parameters["refinements"]:
                nb_cells_ref = refinement["cells"]
                refinement_point = refinement["x"]
                print("Mesh size before local refinement is " +
                      str(len(mesh.cells())))
                while len(mesh.cells()) < \
                        initial_number_of_cells + nb_cells_ref:
                    cell_markers = MeshFunction(
                        "bool", mesh, mesh.topology().dim())
                    cell_markers.set_all(False)
                    for cell in cells(mesh):
                        if cell.midpoint().x() < refinement_point:

                            cell_markers[cell] = True
                    mesh = refine(mesh, cell_markers)
                print("Mesh size after local refinement is " +
                      str(len(mesh.cells())))
                initial_number_of_cells = len(mesh.cells())
        else:
            print('No refinement parameters found')
        return mesh

    def solubility(self, S_0, E_S, k_B, T):
        return S_0*exp(-E_S/k_B/T)

    def solubility_BC(self, P, S):
        return P**0.5*S

    def adaptative_timestep(self, converged, nb_it, dt, dt_min,
                            stepsize_change_ratio, t, t_stop,
                            stepsize_stop_max):
        '''
        Adapts the stepsize as function of the number of iterations of the
        solver.
        Arguments:
        - converged : bool, determines if the time step has converged.
        - nb_it : int, number of iterations
        - dt : Constant(), fenics object
        - dt_min : float, stepsize minimum value
        - stepsize_change_ration : float, stepsize change ratio
        - t : float, time
        - t_stop : float, time where adaptative time step stops
        - stepsize_stop_max : float, maximum stepsize after stop
        Returns:
        - dt : Constant(), fenics object
        '''
        while converged is False:
            dt.assign(float(dt)/stepsize_change_ratio)
            nb_it, converged = solver.solve()
            if float(dt) < dt_min:
                sys.exit('Error: stepsize reached minimal value')
        if t > t_stop:
            if float(dt) > stepsize_stop_max:
                dt.assign(stepsize_stop_max)

        else:
            if nb_it < 5:
                dt.assign(float(dt)*stepsize_change_ratio)
            else:
                dt.assign(float(dt)/stepsize_change_ratio)

        return dt

    def apply_boundary_conditions(self, boundary_conditions, V,
                                  surface_marker, ds):
        '''
        Create a list of DirichletBCs.
        Arguments:
        - boundary_conditions: dict, parameters for bcs
        - V: FunctionSpace,
        - surface_marker: MeshFunction, contains the markers for
        the different surfaces
        - ds: Measurement
        Returns:
        - bcs: list, contains fenics DirichletBC
        - expression: list, contains the fenics Expression
        to be updated.
        '''
        bcs = list()
        expressions = list()
        for type_BC in boundary_conditions:
            for BC in boundary_conditions[type_BC]:
                if type_BC == "dc":
                    value_BC = Expression(str(BC['value']), t=0, degree=2)
                elif type_BC == "solubility":
                    pressure = BC["pressure"]
                    value_BC = self.solubility_BC(
                        pressure, BC["density"]*self.solubility(
                            BC["S_0"], BC["E_S"],
                            k_B, temp(0)))
                    value_BC = Expression(sp.printing.ccode(value_BC), t=0,
                                          degree=2)
                expressions.append(value_BC)
                if type(BC['surface']) == list:
                    for surface in BC['surface']:
                        bci = DirichletBC(V.sub(0), value_BC,
                                          surface_marker, surface)
                        bcs.append(bci)
                else:
                    bci = DirichletBC(V.sub(0), value_BC,
                                      surface_marker, BC['surface'])
                    bcs.append(bci)

        return bcs, expressions

    def update_bc(self, expressions, t):
        '''
        Arguments:
        - expressions: list, contains the fenics Expression
        to be updated.
        - t: float, time.
        Update all FEniCS Expression() in expressions.
        '''
        for expression in expressions:
            expression.t = t
        return expressions


class myclass(Ttrap):
    def __init__(self):
        ttrap = Ttrap()

        def define_boundary_conditions():
            '''
            Returns a dict that contains the parameters for
            boundary conditions.
            Parameters needed :
            - dc : "surface" , "value"
            - solubility : "surface", "S_0", "E_S", "pressure", "density"
            '''
            x, y, z, t = sp.symbols('x[0] x[1] x[2] t')
            boundary_conditions = {
                "dc": [
                    {
                        "surface": [1],
                        "value": 0
                        },
                    {
                        "surface": [2],
                        "value": 0
                        }
                ],
                "solubility": [
                    ]
            }
            return boundary_conditions

        def define_materials():
            '''
            Creates a list of dicts corresponding to the different materials
            and containing properties.
            Returns:
            -materials : list of dicts corresponding to the different materials
            and containing properties.
            '''
            materials = []
            material1 = {
                "alpha": 1.1e-10,  # lattice constant ()
                "beta": 6*6.3e28,  # number of solute sites per atom (6 for W)
                "density": 6.3e28,
                "borders": [0, 20e-6],
                "E_diff": 0.39,
                "D_0": 4.1e-7,
                "id": 1
            }
            materials = [material1]
            return materials

        self.__mesh_parameters = {
            "initial_number_of_cells": 200,
            "size": 20e-6,
            "refinements": [
                {
                    "cells": 300,
                    "x": 3e-6
                },
                {
                    "cells": 120,
                    "x": 30e-9
                }
            ],
            }
        self.__mesh = ttrap.mesh_and_refine(self.__mesh_parameters)
        self.__materials = define_materials()
        self.__BC = define_boundary_conditions()

    def define_exports(self):
        exports = {
            "txt": {
                "functions": ['retention'],
                "times": [100],
                "labels": ['retention']
            },
            "xdmf": {
                "functions": ['solute', '1', '2', '3', '4', '5', 'retention'],
                "labels":  ['solute', 'trap_1', 'trap_2',
                            'trap_3', 'trap_4', 'trap_5', 'retention']
            },
            "TDS": {
                "label": "desorption",
                "TDS_time": 450
                }
        }
        folder = "Solution/"
        return folder, exports

    def define_traps(self):
        '''
        Create a list of dicts corresponding to the different traps
        and containing properties.
        Arguments:
        - n_trap_3_ : Function(W), only required if extrinsic trap is
        simulated.
        Returns:
        -materials : list of dicts corresponding to the different traps
        and containing properties.
        '''
        traps = [
            {
                "energy": 0.87,
                "density": 1.3e-3*6.3e28,
                "materials": [1]
            },
            {
                "energy": 1.0,
                "density": 4e-4*6.3e28,
                "materials": [1]
            },
            {
                "energy": 1.5,
                "density": n_trap_3,
                "materials": [1]
            },
            {
                "energy": 1.4,
                "density": 0*2e-4*6.3e28,
                "materials": [1]
            },
            {
                "energy": 1.4,
                "density": 0,
                "materials": [1]
            }
        ]
        return traps

    def define_solving_parameters(self):
        '''
        Returns the solving parameters needed for simulation.
        '''
        solving_parameters = {
            "final_time": self.implantation_time +
            self.resting_time+self.TDS_time,
            "num_steps": 2*int(self.implantation_time +
                               self.resting_time+self.TDS_time),
            "adaptative_time_step": {
                "stepsize_change_ratio": 1.1,
                "t_stop": self.implantation_time + self.resting_time - 20,
                "stepsize_stop_max": 0.5,
                "dt_min": 1e-5
                },
            "newton_solver": {
                "absolute_tolerance": 1e10,
                "relative_tolerance": 1e-9,
                "maximum_it": 50,
            }
        }
        return solving_parameters

    def define_temperature(self):
        '''
        Returns the temperature sequence as function of space and time
        returns: T, Expression()
        '''
        x, y, z, t = sp.symbols('x[0] x[1] x[2] t')
        T = {
            'type': "expression",
            'value': sp.printing.ccode(
                300 + (t > self.implantation_time+self.resting_time) *
                self.ramp * (t - (self.implantation_time+self.resting_time)))
        }
        return T

    def define_source_term(self):
        '''
        Returns:
        - source_term, dict. Contains flux and distribution.
        '''
        x, y, z, t = sp.symbols('x[0] x[1] x[2] t')
        center = 4.5e-9  # + 20e-9
        width = 2.5e-9
        r = 0
        flux = (1-r)*2.5e19 * (t <= self.implantation_time)
        distribution = 1/(width*(2*3.14)**0.5) * \
            sp.exp(-0.5*((x-center)/width)**2)
        source_term = {
            'flux': sp.printing.ccode(flux),
            'distribution': sp.printing.ccode(distribution)
            }
        return source_term

    def getMesh(self):
        return self.__mesh

    def getExports(self):
        return self.__exports

    def getMaterials(self):
        return self.__materials

    def getMeshParameters(self):
        return self.__mesh_parameters

    def getBC(self):
        return self.__BC

    # Declaration of variables
    implantation_time = 400
    resting_time = 50
    ramp = 8
    delta_TDS = 500
    TDS_time = int(delta_TDS / ramp) + 1
    n_trap_3a_max = 1e-1*6.3e28
    n_trap_3b_max = 1e-2*6.3e28
    rate_3a = 6e-4
    rate_3b = 2e-4
    xp = 1e-6
    v_0 = 1e13  # frequency factor s-1
    k_B = 8.6e-5  # Boltzmann constant


ttrap = myclass()

n_trap_3a_max = ttrap.n_trap_3a_max
n_trap_3b_max = ttrap.n_trap_3b_max
rate_3a = ttrap.rate_3a
rate_3b = ttrap.rate_3b
xp = ttrap.xp
v_0 = ttrap.v_0  # frequency factor s-1
k_B = ttrap.k_B  # Boltzmann constant
solving_parameters = ttrap.define_solving_parameters()
Time = solving_parameters["final_time"]
num_steps = solving_parameters["num_steps"]
dT = Time / num_steps
dt = Constant(dT)  # time step size
t = 0  # Initialising time to 0s
stepsize_change_ratio = solving_parameters[
    "adaptative_time_step"][
        "stepsize_change_ratio"]
t_stop = solving_parameters["adaptative_time_step"]["t_stop"]
stepsize_stop_max = solving_parameters[
    "adaptative_time_step"][
        "stepsize_stop_max"]
dt_min = solving_parameters["adaptative_time_step"]["dt_min"]
size = ttrap.getMeshParameters()["size"]
# Mesh and refinement
materials = ttrap.getMaterials()
mesh = ttrap.getMesh()

# Define function space for system of concentrations and properties
P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1, P1, P1, P1, P1])
V = FunctionSpace(mesh, element)
W = FunctionSpace(mesh, 'P', 1)


# Define and mark subdomains
volume_markers, dx, surface_markers, ds = ttrap.subdomains(mesh, materials)

# Define expressions used in variational forms
print('Defining source terms')
source_term = ttrap.define_source_term()
teta = Expression('(x[0] < xp && x[0] > 0)? 1/xp : 0',
                  xp=xp, degree=1)

flux_ = Expression(source_term["flux"], t=0, degree=2)
f = Expression(source_term["distribution"], t=0, degree=2)
T = ttrap.define_temperature()
temp = Expression(T['value'], t=t, degree=2)

# BCs
print('Defining boundary conditions')


bcs, expressions = ttrap.apply_boundary_conditions(
    ttrap.getBC(), V, surface_markers, ds)


# Define test functions
v_1, v_2, v_3, v_4, v_5, v_6 = TestFunctions(V)
testfunctions = [v_1, v_2, v_3, v_4, v_5, v_6]
v_trap_3 = TestFunction(W)


u = Function(V)
du = TrialFunction(V)
n_trap_3 = Function(W)  # trap 3 density


# Split system functions to access components
u_1, u_2, u_3, u_4, u_5, u_6 = split(u)
solutions = [u_1, u_2, u_3, u_4, u_5, u_6]

print('Defining initial values')
ini_u = Expression(("0", "0", "0", "0", "0", "0"), degree=1)
u_n = interpolate(ini_u, V)
u_n1, u_n2, u_n3, u_n4, u_n5, u_n6 = split(u_n)
previous_solutions = [u_n1, u_n2, u_n3, u_n4, u_n5, u_n6]

ini_n_trap_3 = Expression("0", degree=1)
n_trap_3_n = interpolate(ini_n_trap_3, W)
n_trap_3_ = Function(W)


print('Defining variational problem')

# Define variational problem1
traps = ttrap.define_traps()
F = ttrap.formulation(traps, solutions, testfunctions, previous_solutions)

F_n3 = ((n_trap_3 - n_trap_3_n)/dt)*v_trap_3*dx
F_n3 += -flux_*(
    (1 - n_trap_3_n/n_trap_3a_max)*rate_3a*f +
    (1 - n_trap_3_n/n_trap_3b_max)*rate_3b*teta) \
    * v_trap_3*dx

# Solution files
folder, exports = ttrap.define_exports()

filedesorption = ttrap.save_as()  # folder + exports["TDS"]["label"]+ '.csv'
files = ttrap.define_xdmf_files(exports, folder)
#  Time-stepping
print('Time stepping...')
total_n = 0
desorption = list()
export_total = list()
set_log_level(30)  # Set the log level to WARNING
#set_log_level(20) # Set the log level to INFO

timer = Timer()  # start timer

while t < Time:

    print(str(round(t/Time*100, 2)) + ' %        ' + str(round(t, 1)) + ' s' +
          "    Ellapsed time so far: %s s" % round(timer.elapsed()[0], 1),
          end="\r")

    J = derivative(F, u, du)  # Define the Jacobian
    problem = NonlinearVariationalProblem(F, u, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    solver.parameters["newton_solver"]["absolute_tolerance"] = \
        solving_parameters['newton_solver']['absolute_tolerance']
    solver.parameters["newton_solver"]["relative_tolerance"] = \
        solving_parameters['newton_solver']['relative_tolerance']
    nb_it, converged = solver.solve()
    dt = ttrap.adaptative_timestep(converged=converged, nb_it=nb_it, dt=dt,
                                   stepsize_change_ratio=stepsize_change_ratio,
                                   dt_min=dt_min, t=t, t_stop=t_stop,
                                   stepsize_stop_max=stepsize_stop_max)

    solve(F_n3 == 0, n_trap_3, [])

    _u_1, _u_2, _u_3, _u_4, _u_5, _u_6 = u.split()
    res = [_u_1, _u_2, _u_3, _u_4, _u_5, _u_6]
    retention = project(_u_1)
    total_trap = 0
    for i in range(1, len(traps)+1):
        sol = res[i]
        total_trap += assemble(sol*dx)
        retention = project(retention + res[i], W)
    ttrap.export_xdmf([_u_1, _u_2, _u_3, _u_4, _u_5, _u_6, retention], exports,
                      files, t)
    dt = ttrap.export_profiles([_u_1, _u_2, _u_3, _u_4, _u_5, _u_6, retention],
                               exports, t, dt)

    total_sol = assemble(_u_1*dx)
    total = total_trap + total_sol
    desorption_rate = [-(total-total_n)/float(dt), temp(size/2), t]
    total_n = total
    if t > ttrap.implantation_time+ttrap.resting_time:
        desorption.append(desorption_rate)

    # Update previous solutions
    u_n.assign(u)
    n_trap_3_n.assign(n_trap_3)
    # Update current time
    t += float(dt)
    temp.t += float(dt)
    flux_.t += float(dt)
    expressions = ttrap.update_bc(expressions, t)


ttrap.export_TDS(filedesorption, desorption)
print('\007s')
