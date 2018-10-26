from fenics import *
from fenics import *
from dolfin import *
import numpy as np
import csv
import sys
import os
import argparse

implantation_time=100.0
resting_time=0
TDS_time=0
Time =implantation_time+resting_time+TDS_time # final time
num_steps = 8*int(implantation_time+resting_time+TDS_time) # number of time steps
k = Time / num_steps # time step size
dt = Constant(k)
t=0 #Initialising time to 0s
size=5e-6
mesh = IntervalMesh(6000, 0, size)

# Define function space for system of concentrations
P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1, P1, P1])
V = FunctionSpace(mesh, element)

# BCs
print('Defining boundary conditions')
def boundary(x, on_boundary):
    return on_boundary and (near(x[0],0) or near(x[0],size))
##Tritium concentration
inside_bc_c=Expression(('0','0','0','0'), t=0, degree=1) 
bci_c = DirichletBC(V,inside_bc_c,boundary)
bcs = [bci_c]


# Define test functions
v_1, v_2, v_3, v_4 = TestFunctions(V)

u = Function(V)
u_n = Function(V)

# Split system functions to access components
u_1, u_2, u_3, u_4 = split(u)

print('Defining initial values')
ini = Expression(("x[0]<1e-6 ? 0 : 0","0","0", "0"), degree=1)
u_n = interpolate(ini, V)
u_n1, u_n2, u_n3, u_n4 = split(u_n)

print('Defining source terms')
#f = Expression("x[0]<10e-9 ? 1 : 0", degree=1)
f = Expression('t<implantation_time ? 2e16/(6.3e28) * 1/(2.5e-9*2*3.14) *exp(-0.5*pow(((x[0]-4.5e-9)/2.5e-9), 2)): 0',implantation_time=implantation_time,e=size,t=0,degree=2)#  This is the tritium volumetric source term   -1/(1/3*e*pow(2*3.14,0.5))*exp(-0.5*(x[0]/pow(1/3*e,2)))

# Define expressions used in variational forms
print('Defining variational problem')

density = 6.3e28
n_trap_1 = 1e-3 # trap 1 density
n_trap_2 = 4e-4 # trap 2 density
E1=0.87 # in eV trap 1 activation energy
E2=1.0 # in eV activation energy
alpha = Constant(1.1e-10) # lattice constant ()
beta = Constant(6) # number of solute sites per atom (6 for W)
v_0=1e13 #frequency factor s-1
#n_trap = Constant(n_trap)
k_B = 8.6e-5
temp=Expression('t < (implantation_time+resting_time) ? 300 : 300+8*(t-(implantation_time+resting_time))',implantation_time=implantation_time,resting_time=resting_time,t=0,degree=2)

def T_var(t):
  if t<implantation_time: 
    return 300 
  elif t<implantation_time+resting_time: 
    return 300
  else:
     return 300+8*(t-(implantation_time+resting_time))
def calculate_D(T,subdomain):
    return 1.38e-7*exp(-0.2/(k_B*T))
D = calculate_D(T_var(0), 0)




# Define variational problem
F = ((u_1 - u_n1) / dt)*v_1*dx + D*dot(grad(u_1), grad(v_1))*dx  + ((u_2 - u_n2) / dt)*v_1*dx - f*v_1*dx\
    + ((u_2 - u_n2) / dt)*v_2*dx - D/alpha/alpha/beta*u_1*(n_trap_1 - u_2)*v_2*dx + v_0*exp(-E1/k_B/temp)*u_2*v_2*dx \
    + ((u_3 - u_n3) / dt)*v_3*dx - D/alpha/alpha/beta*u_1*(n_trap_2 - u_3)*v_3*dx + v_0*exp(-E2/k_B/temp)*u_3*v_3*dx



vtkfile_u_1 = File('Solution/c_sol.pvd')
vtkfile_u_2 = File('Solution/c_trap1.pvd')
vtkfile_u_3 = File('Solution/c_trap2.pvd')
filedesorption="desorption.csv"
print('Time stepping')
# Time-stepping
t = 0
desorption=list()
total_n=0
for n in range(num_steps):
    # Update current time
    t += k
    temp.t += k
    D=calculate_D(T_var(t),0)
    print(round(t/Time*100,5), "%")
    print(round(t,4), "s")
    #print(t)
    # Solve variational problem for time step
    solve(F == 0, u, bcs, solver_parameters={"newton_solver":{"absolute_tolerance":1e-19}})


    _u_1, _u_2, _u_3, _u_4 = u.split()
    #print("Sol",_u_1(0.5e-9))
    #print("Trap1",_u_2(0.5e-9))
    #print("Trap2",_u_3(0.5e-9))
    #print("Sol",_u_1(1.5e-6))
    #print("Trap1",_u_2(1.5e-6))
    #print("Trap2",_u_3(1.5e-6))
    
    # Save solution to file (VTK)
    vtkfile_u_1 << (_u_1, t)
    vtkfile_u_2 << (_u_2, t)
    vtkfile_u_3 << (_u_3, t)

    total_trap1=assemble(_u_2*density*dx)
    total_trap2=assemble(_u_3*density*dx)
    total_trap=total_trap1+total_trap2
    total_sol=assemble(_u_1*density*dx)
    total=total_sol+total_trap
    desorption_rate=[-(total-total_n)/k,T_var(t-k),t]
    total_n=total
  
    if t>implantation_time+resting_time:
        desorption.append(desorption_rate)
        #print('Total of D soluble = ' + str(total_sol))
        #print('Total of D traped 1= ' + str(total_trap1))
        #print('Total of D traped 2= ' + str(total_trap2))
        #print("Total of D = "+str(total))
        #print("Desorption rate = " + str(desorption_rate))

    # Update previous solutions
    u_n.assign(u)

with open(filedesorption, "w+") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(['dTt'])
    for val in desorption:
        writer.writerows([val])