from fenics import *
from parameters import parameters, id_W, id_Cu, id_CuCrZr, \
    atom_density_Cu, atom_density_W, atom_density_CuCrZr
from context import FESTIM
from FESTIM.meshing import read_subdomains_from_xdmf
from FESTIM import k_B

# import mesh

mesh = Mesh()
XDMFFile("maillages/Mesh 10/mesh_domains.xdmf").read(mesh)

# Create XDMFFile
folder = 'results/cas_test_simplifie_ITER/4_traps'
labels = ["solute", "1", "2", "3", "4", "retention", "T"]
files_names = ["solute", "1", "2", "3", "4", "retention", "T"]
files = []
for i in range(0, len(labels)):
    files.append(XDMFFile(folder + "/" + files_names[i] + ".xdmf"))


file_difference = XDMFFile(folder + "/test_difference.xdmf")
file_difference.parameters["rewrite_function_mesh"] = False
file_difference.parameters["flush_output"] = True

# Create finite elements and function spaces

P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
DG1 = FiniteElement("DG", mesh.ufl_cell(), 1)
DG0 = FiniteElement("DG", mesh.ufl_cell(), 0)
V_P1 = FunctionSpace(mesh, P1)
V_DG1 = FunctionSpace(mesh, DG1)
V_DG0 = FunctionSpace(mesh, DG0)

# Post treatment

T = Function(V_P1)
solute = Function(V_P1)
trap_1 = Function(V_DG1)
trap_2 = Function(V_DG1)
trap_3 = Function(V_DG1)
trap_4 = Function(V_DG1)
retention = Function(V_DG1)


functions = [solute, trap_1, trap_2, trap_3, trap_4, retention, T]


for i in range(0, 150):
    print(i)
    t = i  # TODO: change this....

    # Read
    for j in range(0, len(functions)):
        files[j].read_checkpoint(functions[j], labels[j], i)
    # Compute the difference
    # This needs to be adapted
    difference = project((solute - retention)/solute, V_DG1)
    # Write
    difference.rename("relative_difference", "relative_difference")
    file_difference.write(difference, t)
