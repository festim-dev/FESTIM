import fenics as f

mesh_1D = f.UnitIntervalMesh(10)
V_0D = f.FunctionSpace(mesh_1D, "R", 0)
c_0D = f.interpolate(f.Constant(1), V_0D)

V_1D = f.FunctionSpace(mesh_1D, "P", 1)
c_1D = f.interpolate(f.Expression("x[0]", degree=1), V_1D)

mesh_2D = f.UnitSquareMesh(10, 10)
V_2D = f.FunctionSpace(mesh_2D, "P", 1)
c_2D = f.interpolate(f.Expression("x[0]", degree=1), V_2D)

mesh_3D = f.UnitCubeMesh(10, 10, 10)
V_3D = f.FunctionSpace(mesh_3D, "P", 1)
c_3D = f.interpolate(f.Expression("x[0]", degree=1), V_3D)
