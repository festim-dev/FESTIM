=============
Newton solver
=============

For advanced simulations, the parameters of the Newton solver can be adapted depending on a specific problem. 

-----------------
Built-in options
-----------------
A limited set of the solver parameters can be accessed via the built-in attributes of classes. 

The parameters of the Newton solver for :class:`festim.HTransportProblem` can be chosen in :class:`festim.Settings` (see :ref:`settings_ug`). Absolute and relative tolerances of the Newton solver
are defined with ``absolute_tolerance`` and ``relative_tolerance`` attributes, respectively. The maximum number of the solver iterations can be set using 
the ``maximum_iterations`` parameter. Additionally, there is an option to choose linear solver and preconditioning methods that may be more suitable for particular problems.

The linear solver method can be set with the ``linear_solver`` attribute. The list of available linear solvers can be viewed with: ``print(fenics.list_linear_solver_methods())``.

.. dropdown:: Linear solver methods

    * "bicgstab" - Biconjugate gradient stabilized method
    * "cg" - Conjugate gradient method
    * "gmres" - Generalized minimal residual method
    * "minres" - Minimal residual method
    * "mumps" - MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)
    * "petsc" - PETSc built in LU solver
    * "richardson" - Richardson method 
    * "superlu" - SuperLU
    * "superlu_dist" - Parallel SuperLU
    * "tfqmr" - Transpose-free quasi-minimal residual method
    * "umfpack" - UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)

The preconditioner can be set with the ``preconditioner`` attribute. The list of available preconditioners can be viewed with: ``print(fenics.list_krylov_solver_preconditioners())``.

.. dropdown:: Preconditioners

    * "amg" - Algebraic multigrid
    * "hypre_amg" - Hypre algebraic multigrid (BoomerAMG)
    * "hypre_euclid" - Hypre parallel incomplete LU factorization
    * "hypre_parasails" - Hypre parallel sparse approximate inverse
    * "icc" - Incomplete Cholesky factorization
    * "ilu" - Incomplete LU factorization
    * "jacobi" - Jacobi iteration 
    * "petsc_amg" - PETSc algebraic multigrid
    * "sor" - Successive over-relaxation

Similarly, the Newton solver parameters of :class:`festim.HeatTransferProblem`, :class:`festim.ExtrinsicTrap`, or :class:`festim.NeutronInducedTrap` 
can be defined if needed. Here is an example for the heat transfer problem:

.. testsetup::

    import fenics
    import festim as F

    model = F.Simulation()
    model.mesh = F.MeshFromVertices([1, 2, 3, 4, 5])
    model.materials = F.Material(id=1, D_0=1, E_D=0, thermal_cond=10, rho=2, heat_capacity=3)
    model.settings = F.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        final_time=1,
    )
    model.dt = F.Stepsize(1)

.. testcode::

    from festim import HeatTransferProblem

    model.T = HeatTransferProblem(
        transient=True,
        initial_condition=300,
        absolute_tolerance=1.0,
        relative_tolerance=1e-10,
        maximum_iterations=50,
        linear_solver="gmres",
        preconditioner="icc",
        )



--------------
Custom solver
--------------

For a finer control, the built-in Newton solver can be overwritten with a custom solver based on the ``fenics.NewtonSolver`` class.

.. warning::
    
    Defining a custom Newton solver will override the solver parameters given with the built-in settings.

A user-defined Newton solver can be provided after :class:`festim.Simulation.initialise()`. Here is a simple example for the H transport problem:

.. testsetup:: custom_solver_simple

    import fenics
    import festim as F

    model = F.Simulation()
    model.T = 500
    model.mesh = F.MeshFromVertices([1, 2, 3, 4, 5])
    model.materials = F.Material(id=1, D_0=1, E_D=0)
    model.settings = F.Settings(
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        transient=False,
    )

.. testcode:: custom_solver_simple

    import fenics

    custom_solver = fenics.NewtonSolver()
    custom_solver.parameters["error_on_nonconvergence"] = False
    custom_solver.parameters["absolute_tolerance"] = 1e10
    custom_solver.parameters["relative_tolerance"] = 1e-10
    custom_solver.parameters["maximum_iterations"] = 100
    custom_solver.parameters["linear_solver"] = "gmres"
    custom_solver.parameters["preconditioner"] = "ilu"

    model.initialise()

    model.h_transport_problem.newton_solver = custom_solver

    model.run()

.. testoutput:: custom_solver_simple
   :options: +ELLIPSIS
   :hide:

   ...

.. warning::
    
    For a stationary heat transfer problem, a custom Newton solver has to be provided before the simulation initialisation! 

To extend the functionality, the `NewtonSolver <https://bitbucket.org/fenics-project/dolfin/src/master/dolfin/nls/NewtonSolver.cpp>`_ class 
can be overwritten: 

.. testcode::

    import fenics

    class CustomSolver(fenics.NewtonSolver):
        def __init__(self):
            super().__init__()

        def converged(self, r, problem, iteration):
            if iteration == 0:
                self.r0 = r.norm("l2")
            print(f"Iteration {iteration}, relative residual {r.norm('l2')/self.r0}")
            return super().converged(r, problem, iteration)

In this example, the relative residual will be printed after each Newton solver iteration.