from fenics import *
from parameters import parameters, id_W, id_Cu, id_CuCrZr, \
    atom_density_Cu, atom_density_W, atom_density_CuCrZr
from context import FESTIM
from FESTIM.meshing import read_subdomains_from_xdmf
from FESTIM import k_B

# import mesh

mesh = Mesh()
XDMFFile("maillages/Mesh 10/mesh_domains.xdmf").read(mesh)

# import mesh functions
vm, sm = read_subdomains_from_xdmf(
    mesh,
    volumetric_file="maillages/Mesh 10/mesh_domains.xdmf",
    boundary_file="maillages/Mesh 10/mesh_boundaries.xdmf")


# Create XDMFFile
folder = 'results/cas_test_simplifie_ITER/4_traps'
labels = ["solute", "1", "2", "3", "4", "retention", "T"]
files_names = ["solute", "1", "2", "3", "4", "retention", "T"]
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


# # # # # Option 1 : all solutions as a vector.
# # # # # Seems to interpolate on P1 FunctionSpace ... not recommended

# export_file = XDMFFile(folder + "/test_export.xdmf")
# export_file.parameters["flush_output"] = True
# export_file.parameters["rewrite_function_mesh"] = False

# number_of_traps = 4
# # [solute, trap1, .., trap4, retention, T]
# element = [P1] + (number_of_traps + 1)*[DG1] + [P1]
# V_mixed = FunctionSpace(mesh, MixedElement(element))
# u = Function(V_mixed)


# for i in range(0, 150):
#     print(i)
#     t = i  # TODO: change this....

#     # Read
#     for j in range(0, len(element)):
#         component = Function(V_mixed.sub(j).collapse())
#         files[j].read_checkpoint(component, labels[j], i)
#         component = interpolate(component, V_mixed.sub(j).collapse())
#         assign(u.sub(j), component)

#     # Write
#         export_file.write(u, t)


# # # # # Option 2 : not as a vector (visualisation issue in Paraview though)
# # # # # not recommended in current state

# export_file = XDMFFile(folder + "/test_export2.xdmf")
# export_file.parameters["flush_output"] = True
# export_file.parameters["rewrite_function_mesh"] = False


# T = Function(V_P1)
# solute = Function(V_P1)
# trap_1 = Function(V_DG1)
# trap_2 = Function(V_DG1)
# trap_3 = Function(V_DG1)
# trap_4 = Function(V_DG1)
# retention = Function(V_DG1)


# functions = [solute, trap_1, trap_2, trap_3, trap_4, retention, T]
# append = False
# for i in range(0, 5):
#     print(i)
#     t = i  # TODO: change this....
#     for j in range(0, len(labels)):
#         # Read
#         files[j].read_checkpoint(functions[j], labels[j], i)
#         # Write
#         export_file.write_checkpoint(
#             functions[j], labels[j], t,
#             XDMFFile.Encoding.HDF5, append=append)
#         append = True

# # # # # Option 3 : seperate files (only post processed fields)

T = Function(V_P1)
solute = Function(V_P1)
trap_1 = Function(V_DG1)
trap_2 = Function(V_DG1)
trap_3 = Function(V_DG1)
trap_4 = Function(V_DG1)
retention = Function(V_DG1)


functions = [solute, trap_1, trap_2, trap_3, trap_4, retention, T]


# create files
file_solute_atfr = XDMFFile(folder + "/solute_atfr.xdmf")
file_solute_atfr.parameters["rewrite_function_mesh"] = False
file_solute_atfr.parameters["flush_output"] = True
file_rho = XDMFFile(folder + "/rho.xdmf")
file_rho.parameters["rewrite_function_mesh"] = False
file_rho.parameters["flush_output"] = True
file_D = XDMFFile(folder + "/D.xdmf")
file_D.parameters["rewrite_function_mesh"] = False
file_D.parameters["flush_output"] = True

# properties should belong to DG0 functionspace
rho = Function(V_DG0)  # Create function rho (atomic density in at.m-3)
D_0 = Function(V_DG0)  # Create function D_0
E_diff = Function(V_DG0)  # Create function  E_diff

# create properties functions
for cell in cells(mesh):  # Iterate through mesh cells
    subdomain_id = vm[cell]
    if subdomain_id == id_W:
        value_rho = atom_density_W
        value_D_0 = parameters["materials"][0]["D_0"]
        value_E_diff = parameters["materials"][0]["E_diff"]
    elif subdomain_id == id_Cu:
        value_rho = atom_density_Cu
        value_D_0 = parameters["materials"][1]["D_0"]
        value_E_diff = parameters["materials"][1]["E_diff"]
    elif subdomain_id == id_CuCrZr:
        value_rho = atom_density_CuCrZr
        value_D_0 = parameters["materials"][2]["D_0"]
        value_E_diff = parameters["materials"][2]["E_diff"]

    # attribute value to the function
    rho.vector()[cell.index()] = value_rho
    D_0.vector()[cell.index()] = value_D_0
    E_diff.vector()[cell.index()] = value_E_diff

file_rho.write(rho)

for i in range(0, 150):
    print(i)
    t = i  # TODO: change this....

    # Read
    for j in range(0, len(functions)):
        files[j].read_checkpoint(functions[j], labels[j], i)
    # Compute
    solute_atfr = project(solute/rho, V_DG1)  # solute concentration in at.fr.
    solute_atfr.rename("solute_atfr", "solute_atfr")  # mandatory
    D = project(D_0*exp(-E_diff/k_B/T), V_DG1)  # diffusion coefficient in m2.s-1
    D.rename("D", "D")
    # Write
    file_solute_atfr.write(solute_atfr, t)
    file_D.write(D, t)
