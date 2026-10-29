===================
Running in parallel
===================

FESTIM can be run in parallel on N cores using the command: 

.. code::
    
    mpirun -np N python your_FESTIM_script.py


The mesh is partitioned between the processes by DOLFINx and each process solves its share of the problem.
The quantities computed by the exports (the ``value`` attribute of :class:`festim.SurfaceFlux`, :class:`festim.TotalVolume`, :class:`festim.MaximumVolume`...) are global: they are the same on every process.
On the other hand, the arrays of the solution (``species.solution.x.array``) only hold the degrees of freedom owned by the process, so any post-processing done directly in the script needs an MPI reduction, for example ``mesh.comm.allreduce(local_max, op=MPI.MAX)``.

.. warning::

    Not every feature of FESTIM supports parallel runs yet. In particular:

    * Codimensional (manifold) subdomains lying *inside* the mesh (either manifolds through 
      a single volume subdomain or along the interface between 2+ volume subdomains, see :ref:`Codimensional (manifold) Subdomains`) cannot be used in parallel: ``initialise()``
      raises an error. Manifolds lying on the outer boundary of the domain are not affected.
    * :class:`festim.Profile1DExport` only collects the part of the profile owned by each process, without raising an error.
    * The ``filename`` of derived quantities (:class:`festim.SurfaceFlux`, :class:`festim.TotalVolume`...) is written by every process, so the CSV file contains duplicated rows.
      The ``value`` attribute of these exports is correct.

    When in doubt, compare the results of a small case run in serial and in parallel.