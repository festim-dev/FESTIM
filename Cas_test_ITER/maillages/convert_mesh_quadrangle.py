import meshio
folder = "mesh_1"
name = folder + "/mesh_1"
if name.endswith(".med") is False:
    mesh_file_med = name + ".med"
mesh = meshio.read(mesh_file_med)
#meshio.write(mesh_file_xdmf, mesh)  # won't work for FEniCS, mixed

# In order to use MeshFunction of FEniCS
# The tag must be a positive number (size_t)
mesh.cell_data["quad"]["cell_tags"] *= -1
mesh.cell_data["line"]["cell_tags"] *= -1
# mesh.cell_tags = {-6: ['Down'], -7: ['Top'], -8: ['Lying on Top']}


# Export mesh that contains only triangular faces
# along with tags
meshio.write_points_cells(
    folder + "/mesh_domains.xdmf",
    mesh.points,
    {"quad": mesh.cells["quad"]},
    cell_data={"quad": {"f": mesh.cell_data["quad"]["cell_tags"]}},
)

# Export mesh that contains only lines
# along with tags
meshio.write_points_cells(
    folder + "/mesh_boundaries.xdmf",
    mesh.points,
    {"line": mesh.cells["line"]},
    cell_data={"line": {"f": mesh.cell_data["line"]["cell_tags"]}},
)
