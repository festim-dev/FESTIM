from fenics import *
from dolfin import *
import numpy as np
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
                print("The file " + filedesorption + " is currently busy."
                      "Please close the application then press any key")
                input()
        return

    def calculate_D(self, T, E, D_0):
        '''
        Calculate the diffusion coeff at a certain temperature
        and for a specific material (subdomain)
        Arguments:
        - T : float, temperature
        - E : float, diffusion energy
        - D_0 : float, diffusion pre-exponential factor
        Returns : float, the diffusion coefficient
        '''
        coefficient = D_0 * exp(-E/k_B/T)

        return coefficient

    def update_D(self, mesh, volume_markers, materials, T):
        '''
        Iterates through the mesh and compute the value of D
        Arguments:
        - mesh : the mesh
        - volume_markers : MeshFunction that contains the subdomains
        - T : float, the temperature
        Returns : the Function D
        '''
        D = Function(V0)
        for cell in cells(mesh):
            volume_id = volume_markers[cell]
            found = False
            for material in materials:
                if volume_id == material["id"]:
                    found = True
                    D.vector()[cell.index()] = \
                        self.calculate_D(T, material['E_diff'], material['D_0'])
                    break
            if found is False:
                print('Computing D: Volume ID not found')
        return D

    def update_alpha(self, mesh, volume_markers, materials):
        '''
        Iterates through the mesh and compute the value of D
        Arguments:
        - mesh : the mesh
        - volume_markers : MeshFunction that contains the subdomains
        - materials : list, contains all the materials dictionaries

        Returns : the Function alpha
        '''
        alpha = Function(V0)
        for cell in cells(mesh):
            volume_id = volume_markers[cell]
            found = False
            for material in materials:
                if volume_id == material["id"]:
                    found = True
                    alpha.vector()[cell.index()] = material['alpha']
                    break
            if found is False:
                print('Computing alpha: Volume ID not found')
        return alpha

    def update_beta(self, mesh, volume_markers, materials):
        '''
        Iterates through the mesh and compute the value of D
        Arguments:
        - mesh : the mesh
        - volume_markers : MeshFunction that contains the subdomains
        - materials : list, contains all the materials dictionaries

        Returns : the Function beta
        '''
        beta = Function(V0)
        for cell in cells(mesh):
            volume_id = volume_markers[cell]
            found = False
            for material in materials:
                if volume_id == material["id"]:
                    found = True
                    beta.vector()[cell.index()] = material['beta']
                    break
            if found is False:
                print('Computing beta: Volume ID not found')
        return beta

    def formulation(self, traps, solutions, testfunctions, previous_solutions):
        ''' formulation takes traps as argument (list).
        Parameters:
        - traps : dict, contains the energy, density and domains
        of the traps
        - solutions : list, contains the solution fields
        - testfunctions : list, contains the testfunctions
        - previous_solutions : list, contains the previous solution fields

        Returns:
        - F : variational formulation
        '''
        transient_sol = ((u_1 - u_n1) / dt)*v_1*dx
        diff_sol = D*dot(grad(u_1), grad(v_1))*dx
        source_sol = - (1-r)*flux_*f*v_1*dx

        F = 0
        F += transient_sol + source_sol + diff_sol
        i = 1
        for trap in traps:
            #print(i)
            trap_density = trap['density']
            energy = trap['energy']
            material = trap['materials']
            F += ((solutions[i] - previous_solutions[i]) / dt)*testfunctions[i]*dx
            if type(trap['materials']) is list:
                for subdomain in trap['materials']:
                    F += - D/alpha/alpha/beta*u_1*(trap_density - solutions[i])*testfunctions[i]*dx(subdomain)
                    F += v_0*exp(-energy/k_B/temp)*solutions[i]*testfunctions[i]*dx(subdomain)
            else:
                subdomain = trap['materials']
                F += - D/alpha/alpha/beta*u_1*(trap_density - solutions[i])*testfunctions[i]*dx(subdomain)
                F += v_0*exp(-energy/k_B/temp)*solutions[i]*testfunctions[i]*dx(subdomain)
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

        surface_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
        surface_markers.set_all(0)
        i=0
        for f in facets(mesh):
            i+=1
            x0 = f.midpoint()
            surface_markers[f] = 0
            if near(x0.x(), 0):
                surface_markers[f] = 1
            if near(x0.x(), size):
                surface_markers[f] = 2
        measurement_ds = ds(subdomain_data=surface_markers)
        return volume_markers, measurement_dx, surface_markers, measurement_ds

    def define_traps(self, n_trap_3_):
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
                "energy": 1,
                "density": 1e-2*6.3e28,
                "materials": [1]
            },
            {
                "energy": 1.4,
                "density": 0*1e-2*6.3e28,
                "materials": [1]
            }
            ,
            {
                "energy": 0.9,
                "density": 0*2e-4*6.3e28,#0*n_trap_3_,
                "materials": [1]
            }
            ,
            {
                "energy": 1.4,
                "density":0*2e-4*6.3e28,
                "materials": [1]
            }
            ,
            {
                "energy": 1.4,
                "density": 0,
                "materials": [1]
            }
        ]
        return traps

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
                print("Mesh size before local refinement is " + str(len(mesh.cells())))
                while len(mesh.cells()) < initial_number_of_cells + nb_cells_ref:
                    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
                    cell_markers.set_all(False)
                    for cell in cells(mesh):
                        if cell.midpoint().x() < refinement_point:
                            cell_markers[cell] = True
                    mesh = refine(mesh, cell_markers)
                print("Mesh size after local refinement is " + str(len(mesh.cells())))
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
            #print(float(dt))
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
        '''
        bcs = list()
        for type_BC in boundary_conditions:
            for BC in boundary_conditions[type_BC]:
                if type_BC == "dc":
                    value_BC = Expression(str(BC['value']), t=0, degree=2)
                elif type_BC == "solubility":
                    pressure = Expression(str(BC["pressure"]), t=0, degree=2)
                    value_BC = self.solubility_BC(
                        pressure(0), BC["density"]*self.solubility(
                            BC["S_0"], BC["E_S"],
                            k_B, temp(size/2)))
                if type(BC['surface']) == list:
                    for surface in BC['surface']:
                        bci = DirichletBC(V.sub(0), value_BC,
                                          surface_marker, surface)
                        bcs.append(bci)
                else:
                    bci = DirichletBC(V.sub(0), value_BC,
                                      surface_marker, BC['surface'])
                    bcs.append(bci)

        return bcs


class myclass(Ttrap):
    def __init__(self):
        ttrap = Ttrap()

        def define_boundary_conditions():
            '''
            Returns a dict that contains the parameters for
            boundary conditions.
            ''' 
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
                ]
                #,
                #"solubility": [
                #    {
                #      "surface": [1],
                #      "S_0":1.3e-4,
                #      "E_S":0.34,
                #      "pressure":0.1/101325,
                #      "density": 6.3e28
                #    }
                #    ]
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
                "alpha": Constant(1.1e-10),  # lattice constant ()
                "beta": Constant(6*6.3e28),  # number of solute sites per atom (6 for W)
                "density": 6.3e28,
                "borders": [0, 20e-6],
                "E_diff": 0.39,
                "D_0": 4.1e-7,
                "id": 1
            }
            material2 = {
                "alpha": Constant(1.1e-10),
                "beta": Constant(6*6.3e28),
                "density": 6.3e28,
                "borders": [10e-9, 75e-9],
                "E_diff": 0.39,
                "D_0": 4.1e-7,
                "id": 2
            }
            materials = [material1]#, material2]
            return materials

        self.__mesh_parameters = {
            "initial_number_of_cells": 1600,
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

    def getMesh(self):
        return self.__mesh

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
    r = 0
    flux = 2.5e19
    n_trap_3a_max = 1e-1*6.3e28
    n_trap_3b_max = 1e-2*6.3e28
    rate_3a = 6e-4
    rate_3b = 2e-4
    xp = 1e-6
    v_0 = 1e13  # frequency factor s-1
    k_B = 8.6e-5  # Boltzmann constant
    TDS_time = int(delta_TDS / ramp) + 1
    Time = implantation_time+resting_time+TDS_time
    num_steps = 10*int(implantation_time+resting_time+TDS_time)
    dT = Time / num_steps  # time step size
    t = 0  # Initialising time to 0s
    stepsize_change_ratio = 1.12
    t_stop = implantation_time + resting_time - 20
    stepsize_stop_max = 0.5
    dt_min = 1e-5

ttrap = myclass()

implantation_time = ttrap.implantation_time
resting_time = ttrap.resting_time
ramp = ttrap.ramp
delta_TDS = ttrap.delta_TDS
r = ttrap.r
flux = ttrap.flux  # /6.3e28
n_trap_3a_max = ttrap.n_trap_3a_max
n_trap_3b_max = ttrap.n_trap_3b_max
rate_3a = ttrap.rate_3a
rate_3b = ttrap.rate_3b
xp = ttrap.xp
v_0 = ttrap.v_0  # frequency factor s-1
k_B = ttrap.k_B  # Boltzmann constant
TDS_time = ttrap.TDS_time
Time = ttrap.Time
num_steps = ttrap.num_steps
dt = Constant(ttrap.dT)  # time step size
dT = ttrap.dT
t = ttrap.t  # Initialising time to 0s
stepsize_change_ratio = ttrap.stepsize_change_ratio
t_stop = ttrap.t_stop
stepsize_stop_max = ttrap.stepsize_stop_max
size = ttrap.getMeshParameters()["size"]
dt_min = ttrap.dt_min

# Mesh and refinement
materials = ttrap.getMaterials()
mesh = ttrap.getMesh()

# Define function space for system of concentrations and properties
P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1, P1, P1, P1, P1])
V = FunctionSpace(mesh, element)
W = FunctionSpace(mesh, 'P', 1)
V0 = FunctionSpace(mesh, 'DG', 0)

x = interpolate(Expression('x[0]', degree=1), W)

# Define and mark subdomains
volume_markers, dx, surface_markers, ds = ttrap.subdomains(mesh, materials)



# Define expressions used in variational forms
print('Defining source terms')
center = 4.5e-9 + 20e-9
width = 2.5e-9
f = Expression('1/(width*pow(2*3.14,0.5))*  \
               exp(-0.5*pow(((x[0]-center)/width), 2))',
               degree=2, center=center, width=width)
teta = Expression('(x[0] < xp && x[0] > 0)? 1/xp : 0',
                  xp=xp, degree=1)
flux_ = Expression('t <= implantation_time ? flux : 0',
                   t=0, implantation_time=implantation_time,
                   flux=flux, degree=1)
temp = Expression('t <= (implantation_time+resting_time) ? \
                  373 : 373+ramp*(t-(implantation_time+resting_time))',
                  implantation_time=implantation_time,
                  resting_time=resting_time,
                  ramp=ramp,
                  t=0, degree=2)


# BCs
print('Defining boundary conditions')


def inside(x, on_boundary):
    return on_boundary and (near(x[0], 0))


def outside(x, on_boundary):
    return on_boundary and (near(x[0], size))

bcs = ttrap.apply_boundary_conditions(ttrap.getBC(), V, surface_markers, ds)


# Define test functions
v_1, v_2, v_3, v_4, v_5, v_6 = TestFunctions(V)
testfunctions = [v_1, v_2, v_3, v_4, v_5, v_6]
v_trap_3 = TestFunction(W)


u = Function(V)
du = TrialFunction(V)
n_trap_3 = TrialFunction(W)  # trap 3 density


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

D = ttrap.update_D(mesh, volume_markers, materials, temp(size/2))
alpha = ttrap.update_alpha(mesh, volume_markers, materials)
beta = ttrap.update_beta(mesh, volume_markers, materials)

# Define variational problem
traps = ttrap.define_traps(n_trap_3_)
F = ttrap.formulation(traps, solutions, testfunctions, previous_solutions)

F_n3 = ((n_trap_3 - n_trap_3_n)/dt)*v_trap_3*dx
F_n3 += -(1-r)*flux_*(
    (1 - n_trap_3_n/n_trap_3a_max)*rate_3a*f +
    (1 - n_trap_3_n/n_trap_3b_max)*rate_3b*teta) \
    * v_trap_3*dx

# Solution files
xdmf_u_1 = XDMFFile('Solution/c_sol.xdmf')
xdmf_u_2 = XDMFFile('Solution/c_trap1.xdmf')
xdmf_u_3 = XDMFFile('Solution/c_trap2.xdmf')
xdmf_u_4 = XDMFFile('Solution/c_trap3.xdmf')
xdmf_u_5 = XDMFFile('Solution/c_trap4.xdmf')
xdmf_u_6 = XDMFFile('Solution/c_trap5.xdmf')
xdmf_retention = XDMFFile('Solution/retention.xdmf')
filedesorption = ttrap.save_as()

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
    solver.parameters["newton_solver"]["absolute_tolerance"] = 1e10
    nb_it, converged = solver.solve()
    dt = ttrap.adaptative_timestep(converged=converged, nb_it=nb_it, dt=dt,
                                   stepsize_change_ratio=stepsize_change_ratio,
                                   dt_min=dt_min, t=t, t_stop=t_stop,
                                   stepsize_stop_max=stepsize_stop_max)
    

    #solve(lhs(F_n3) == rhs(F_n3), n_trap_3_, [])
    _u_1, _u_2, _u_3, _u_4, _u_5, _u_6 = u.split()
    res = [_u_1, _u_2, _u_3, _u_4, _u_5, _u_6]
    # Save solution to file (.xdmf)
    _u_1.rename("solute", "label")
    _u_2.rename("trap_1", "label")
    _u_3.rename("trap_2", "label")
    _u_4.rename("trap_3", "label")
    _u_5.rename("trap_4", "label")
    _u_6.rename("trap_5", "label")
    xdmf_u_1.write(_u_1, t)
    xdmf_u_2.write(_u_2, t)
    xdmf_u_3.write(_u_3, t)
    xdmf_u_4.write(_u_4, t)
    xdmf_u_5.write(_u_5, t)
    xdmf_u_6.write(_u_6, t)
    retention = Function(W)
    retention = project(_u_1)
    i = 1
    total_trap = 0
    for trap in traps:
        sol = res[i]
        total_trap += assemble(sol*dx)
        retention = project(retention + res[i], W)
        i += 1
    retention.rename("retention", "label")
    xdmf_retention.write(retention, t)
    total_sol = assemble(_u_1*dx)
    total = total_trap + total_sol
    desorption_rate = [-(total-total_n)/float(dt), temp(size/2), t]
    #export_total.append([assemble(retention*(size-x)**2*dx)*4*3.15, temp(size/2), t])
    total_n = total
    if t > implantation_time+resting_time:
        desorption.append(desorption_rate)
    # Update previous solutions
    u_n.assign(u)
    n_trap_3_n.assign(n_trap_3_)
    # Update current time
    t += float(dt)
    temp.t += float(dt)
    flux_.t += float(dt)
    if t > implantation_time:
        D = ttrap.update_D(mesh, volume_markers, materials, temp(size/2))

ttrap.export_TDS(filedesorption, desorption_rate)
print('\007s')