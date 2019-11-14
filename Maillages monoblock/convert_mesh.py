import meshio
name = "Mesh 6/Mesh 6"
if name.endswith(".med") is False:
    mesh_file_med = name + ".med"
mesh = meshio.read(mesh_file_med)
#meshio.write(mesh_file_xdmf, mesh)  # won't work for FEniCS, mixed

# In order to use MeshFunction of FEniCS
# The tag must be a positive number (size_t)
mesh.cell_data["triangle"]["cell_tags"] *= -1
mesh.cell_data["line"]["cell_tags"] *= -1
# mesh.cell_tags = {-6: ['Down'], -7: ['Top'], -8: ['Lying on Top']}


# Export mesh that contains only triangular faces
# along with tags
meshio.write_points_cells(
    "mesh_domains.xdmf",
    mesh.points,
    {"triangle": mesh.cells["triangle"]},
    cell_data={"triangle": {"f": mesh.cell_data["triangle"]["cell_tags"]}},
)

# Export mesh that contains only lines
# along with tags
meshio.write_points_cells(
    "mesh_boundaries.xdmf",
    mesh.points,
    {"line": mesh.cells["line"]},
    cell_data={"line": {"f": mesh.cell_data["line"]["cell_tags"]}},
)
