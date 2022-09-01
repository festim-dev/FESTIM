===========
Temperature
===========

Definition of a temperature field or problem is essential for hydrogen transport 
and FESTIM as a whole. 

Temperature is defined using the :code:`Temperature` class

.. code-block:: python

    my_temperature = Temperature(value=300)


* Temperature can be defined as a by analytical expression or can be solved explictly using the :code:`HeatTransferProblem` class
* 

---------------------------------------
Temperature as an analytical expression
---------------------------------------

As an analytical expression the temperature can be defined as a constant value or as an expression:

Temperature as a constant value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The temperature can be defined as a constant value in Kelvin (K):

.. code-block:: python

    my_temperature = Temperature(value=300)


Temperature as an expression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* expressions can be in terms of geomteric space depending on the dimension of the problem.
* Or they can be expressed in terms of time for transient h-transport problem 

For example, a 1D, transient h-transport problem can use an analtical expression in the form of:

.. math::

    100 + 2 x +3 t 

would be used in festim as:

.. code-block:: python

    my_temp = Temperature(
        value=100+2*x+3*t
        )

furthermore, sympy expressions can be accepted


--------------------
Heat transfer solver
--------------------

To define the heat trasnfer problem, use the subclass:


.. code-block:: python

    my_temp = HeatTransferProblem()

which has 6 optional arguments:

* :code:`transient`: If True, a transient simulation will be run. Defaults to True
* :code:`initial_value`: The inital value. Only needed if transient is True. Defaults to 0
* :code:`absolute_tolerance`: the absolute tolerance of the newton solver. Defaults to 1e-03
* :code:`relative_tolerance`: the relative tolerance of the newton solver. Defaults to 1e-10
* :code:`maximum_iterations`: the maximum number of iterations allowed for the newton solver to converge. Defulats to 30
* :code:`linear_solver`: linear solver method for the newton solver, options can be veiwed with the command print(list_linear_solver_methods()). If None, the default fenics linear solver will be used ("umfpack"). More information can be found `online <https://fenicsproject.org/pub/tutorial/html/._ftut1017.html/>`_. Defaults to None

further notes:

* varying linear solvers can be useul for larger more complex problems should there be computaional resource issues.
* Following this, boundary conditions for the heat transfer solver will need to be defined. see section x