from festim import PointValue
import fenics as f
import pytest
import logging
import ipyparallel as ipp


@pytest.mark.parametrize("field", ["solute", "T"])
def test_title(field):
    """
    A simple test to check that the title is set
    correctly in festim.PointValue

    Args:
        field (str, int):  the field ("solute", 0, 1, "T", "retention")
    """
    x = 1
    my_value = PointValue(field, x)
    assert my_value.title == f"{field} value at [{x}] ({my_value.export_unit})"


@pytest.mark.parametrize(
    "mesh,x", [(f.UnitIntervalMesh(10), 1), (f.UnitCubeMesh(10, 10, 10), (1, 0, 1))]
)
def test_point_value_compute(mesh, x):
    """Test that the point value export computes the correct value"""
    V = f.FunctionSpace(mesh, "P", 1)
    c = f.interpolate(f.Expression("x[0]", degree=1), V)

    my_value = PointValue("solute", x)
    my_value.function = c

    expected = c(x)
    produced = my_value.compute()
    assert produced == expected


@pytest.fixture(scope="module")
def cluster():
    cluster = ipp.Cluster(engines="mpi", n=2, log_level=logging.ERROR)
    rc = cluster.start_and_connect_sync()
    yield rc
    cluster.stop_cluster_sync()


def test_point_value_compute_parallel(cluster):
    """Testing that RuntimeError due to mesh partitioning is handled"""

    def compute_value():
        from festim import PointValue
        import fenics as f
        import coverage
        import os

        # Initiate coverage tracking for each subprocess
        cov = coverage.Coverage(data_suffix=f"mpi_{os.getpid()}")
        cov.start()

        mesh = f.UnitSquareMesh(10, 10)
        mesh.bounding_box_tree()

        V = f.FunctionSpace(mesh, "P", 1)
        c = f.interpolate(f.Expression("x[0] + x[1]", degree=1), V)

        x = (0, 0)
        my_value = PointValue("solute", x)
        my_value.function = c

        my_value.compute()

        cov.stop()
        cov.save()

    query = cluster[:].apply_async(compute_value)
    query.wait()

    assert query.successful(), query.error
