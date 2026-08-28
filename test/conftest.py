"""Shared pytest configuration.

Under ``mpirun``, every rank runs the whole test session as an independent pytest
process, so pytest hands each rank its own temporary directory (``pytest-17`` on one
rank, ``pytest-18`` on another). Tests that write a file do so with collective MPI
calls -- ADIOS2 and HDF5 both coordinate across ``COMM_WORLD`` -- so ranks writing to
different paths deadlock, each waiting on a file the others never touch.

io4dolfinx solves this per test, by broadcasting rank 0's path where it builds a
filename (``MPI.COMM_WORLD.bcast(tmp_path, root=0)`` in its ``write_function``
fixture). Overriding the two built-in temporary directory fixtures does the same thing
once, so the whole suite is safe to run in parallel without tests having to opt in.
"""

from pathlib import Path

from mpi4py import MPI

import pytest


def _shared_path(path):
    """Broadcast rank 0's temporary directory to every rank.

    ``bcast`` returns only once rank 0 has reached it, which is after pytest created
    the directory on that rank, so no extra barrier is needed before writing.
    """
    return MPI.COMM_WORLD.bcast(path, root=0)


@pytest.fixture
def tmp_path(tmp_path):
    """pytest's `tmp_path`, identical on every MPI rank."""
    return Path(_shared_path(tmp_path))


@pytest.fixture
def tmpdir(tmpdir):
    """pytest's `tmpdir`, identical on every MPI rank."""
    return type(tmpdir)(_shared_path(str(tmpdir)))
