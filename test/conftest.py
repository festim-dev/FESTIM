"""Shared pytest configuration.

Under ``mpirun``, every rank runs the whole test session as an independent pytest
process, so pytest hands each rank its own temporary directory (``pytest-17`` on one
rank, ``pytest-18`` on another). Tests that write a file do so with collective MPI
calls -- ADIOS2 and HDF5 both coordinate across ``COMM_WORLD`` -- so ranks writing to
different paths deadlock, each waiting on a file the others never touch.

Overriding the two built-in temporary directory fixtures to broadcast rank 0's path
makes the whole suite safe to run in parallel, without tests having to opt in.
"""

from pathlib import Path

from mpi4py import MPI

import pytest


def _shared_path(path):
    """Broadcast rank 0's temporary directory to every rank."""
    comm = MPI.COMM_WORLD
    if comm.size == 1:
        return path
    shared = comm.bcast(str(path), root=0)
    # rank 0 has already created the directory; wait for it before anyone writes
    comm.Barrier()
    return shared


@pytest.fixture
def tmp_path(tmp_path):
    """pytest's `tmp_path`, identical on every MPI rank."""
    return Path(_shared_path(tmp_path))


@pytest.fixture
def tmpdir(tmpdir):
    """pytest's `tmpdir`, identical on every MPI rank."""
    return type(tmpdir)(_shared_path(tmpdir))
