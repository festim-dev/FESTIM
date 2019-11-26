from fenics import *
from parameters import parameters, id_W, id_Cu, id_CuCrZr, \
    atom_density_Cu, atom_density_W, atom_density_CuCrZr
from context import FESTIM
from FESTIM.meshing import read_subdomains_from_xdmf
from FESTIM import k_B

# import mesh
mesh = Mesh()
XDMFFile("maillages/Mesh_ITER/mesh_domains.xdmf").read(mesh)

# import mesh functions
vm, sm = read_subdomains_from_xdmf(
    mesh,
    volumetric_file="maillages/Mesh_ITER/mesh_domains.xdmf",
    boundary_file="maillages/Mesh_ITER/mesh_boundaries.xdmf")

# Create XDMFFile
folder = 'results/ITER_case_theta'
labels = ["0", "1", "2", "3", "4", "sum", "T"]
files_names = ["0", "1", "2", "3", "4", "sum", "T"]
files = []
for i in range(0, len(labels)):
    files.append(XDMFFile(folder + "/" + files_names[i] + ".xdmf"))

# Create finite elements and function spaces
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
DG1 = FiniteElement("DG", mesh.ufl_cell(), 1)
DG0 = FiniteElement("DG", mesh.ufl_cell(), 0)
V_P1 = FunctionSpace(mesh, P1)
V_DG1 = FunctionSpace(mesh, DG1)
V_DG0 = FunctionSpace(mesh, DG0)


# # # # # Option 3 : seperate files (only post processed fields)

T = Function(V_P1)
theta = Function(V_P1, name="theta")
trap_1 = Function(V_DG1)
trap_2 = Function(V_DG1)
trap_3 = Function(V_DG1)
trap_4 = Function(V_DG1)
retention = Function(V_DG1)


functions = [theta, trap_1, trap_2, trap_3, trap_4, retention, T]


# create files
file_theta = XDMFFile(folder + "/theta.xdmf")
file_theta.parameters["rewrite_function_mesh"] = False
file_theta.parameters["flush_output"] = True
file_solute = XDMFFile(folder + "/solute_m3.xdmf")
file_solute.parameters["rewrite_function_mesh"] = False
file_solute.parameters["flush_output"] = True
file_solute_atfr = XDMFFile(folder + "/solute_atfr.xdmf")
file_solute_atfr.parameters["rewrite_function_mesh"] = False
file_solute_atfr.parameters["flush_output"] = True
file_rho = XDMFFile(folder + "/rho.xdmf")
file_rho.parameters["rewrite_function_mesh"] = False
file_rho.parameters["flush_output"] = True
file_D = XDMFFile(folder + "/D.xdmf")
file_D.parameters["rewrite_function_mesh"] = False
file_D.parameters["flush_output"] = True
file_S = XDMFFile(folder + "/S.xdmf")
file_S.parameters["rewrite_function_mesh"] = False
file_S.parameters["flush_output"] = True
file_retention = XDMFFile(folder + "/retention.xdmf")
file_retention.parameters["rewrite_function_mesh"] = False
file_retention.parameters["flush_output"] = True
# properties should belong to DG0 functionspace
rho = Function(V_DG0)  # Create function rho (atomic density in at.m-3)
D_0 = Function(V_DG0)  # Create function D_0
E_diff = Function(V_DG0)  # Create function  E_diff
S_0 = Function(V_DG0)
E_S = Function(V_DG0)

# create properties functions
for cell in cells(mesh):  # Iterate through mesh cells
    subdomain_id = vm[cell]
    if subdomain_id == id_W:
        index = 0
        value_rho = atom_density_W
    elif subdomain_id == id_Cu:
        index = 1
        value_rho = atom_density_Cu
    elif subdomain_id == id_CuCrZr:
        index = 2
        value_rho = atom_density_CuCrZr

    value_D_0 = parameters["materials"][index]["D_0"]
    value_E_diff = parameters["materials"][index]["E_diff"]
    value_S_0 = parameters["materials"][index]["S_0"]
    value_E_S = parameters["materials"][index]["E_S"]
    # attribute value to the function
    rho.vector()[cell.index()] = value_rho
    D_0.vector()[cell.index()] = value_D_0
    E_diff.vector()[cell.index()] = value_E_diff

    S_0.vector()[cell.index()] = value_S_0
    E_S.vector()[cell.index()] = value_E_S

file_rho.write(rho)

for i in range(0, 191):
    print(i)
    t = i  # TODO: change this....

    # Read
    for j in range(0, len(functions)):
        files[j].read_checkpoint(functions[j], labels[j], i)
    # Compute
    # diffusion coefficient in m2.s-1
    D = project(D_0*exp(-E_diff/k_B/T), V_DG1)
    D.rename("D", "D")  # mandatory
    # solubilite coefficient in m2.s-1
    S = project(S_0*exp(-E_S/k_B/T), V_DG1)
    S.rename("S", "S")
    # solute concentration in m-3
    solute_m3 = project(S_0*exp(-E_S/k_B/T)*theta, V_DG1)
    solute_m3.rename("solute_m3", "solute_m3")
    # solute concentration in at.fr.
    solute_atfr = project(solute_m3/rho, V_DG1)
    solute_atfr.rename("solute_atfr", "solute_atfr")
    retention = project(solute_m3 + trap_1 + trap_2 + trap_3 + trap_4, V_DG1)
    retention.rename("retention", "retention")
    # Write
    file_solute.write(solute_m3, t)
    file_solute_atfr.write(solute_atfr, t)
    file_D.write(D, t)
    file_S.write(S, t)
    file_retention.write(retention, t)
