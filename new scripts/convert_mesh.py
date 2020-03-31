import meshio

mesh = meshio.read("Mesh_ITER/Mesh_ITER_44722.med")

# Export mesh that contains only triangular faces
# along with tags

mesh.cell_data["cell_tags"][1] *= -1
mesh.cell_data["cell_tags"][0] *= -1

meshio.write_points_cells(
    "Mesh_ITER/mesh_domains_44722.xdmf",
    mesh.points,
    [mesh.cells[1]],
    cell_data={"f": [mesh.cell_data["cell_tags"][1]]},
)

meshio.write_points_cells(
    "Mesh_ITER/mesh_lines_44722.xdmf",
    mesh.points,
    [mesh.cells[0]],
    cell_data={"f": [mesh.cell_data["cell_tags"][0]]},
)
