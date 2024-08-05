.. _sources_guide:

=======
Sources
=======


Volumetric sources
------------------

Volumetric sources can be set in a simulation by using the :class:`festim.Source` class.

.. testcode::

    import festim as F

    my_model = F.Simulation()

    my_model.sources = [
        F.Source(value=1e20, volume=1, field=0),
        F.Source(value=1e19 * F.x, volume=2, field=0),
        ]

For more information, see :class:`festim.Source`.

Implantation flux
-----------------

Hydrogen implanted in a material can be simulated by a Gaussian-shaped volumetric source with the :class:`festim.ImplantationFlux` class.

.. testcode::

    import festim as F

    my_model = F.Simulation()

    my_model.sources = [
        F.ImplantationFlux(
            flux=1e20,
            imp_depth=1e-9,
            width=1e-9,
            volume=1,
            ),
        ]

This class is used in `this tutorial <https://github.com/festim-dev/FESTIM-workshop/blob/main/tasks/task2.ipynb>`_.


Radioactive decay
-----------------

Radioactive decay can be simulated by a volumetric source with the :class:`festim.RadioactiveDecay` class.

.. testcode::

    import festim as F

    my_model = F.Simulation()

    my_model.sources = [
        F.RadioactiveDecay(
            decay_constant=1.78e-9,
            volume=1,
            field="all",
            ),
        ]
