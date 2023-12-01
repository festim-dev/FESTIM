import subprocess
import sys
import pathlib
import pytest
import festim as F

# Get directory of this file
path = pathlib.Path(__file__).resolve().parent

# Build list of demo programs
demos = []
demo_files = list(path.glob('**/*.py'))
for f in demo_files:
    demos.append((f.parent, f.name))


@pytest.mark.serial
@pytest.mark.parametrize("path,name", demos)
def test_demos(path, name):
    ret = subprocess.run([sys.executable, name], cwd=str(path), check=True)
    assert ret.returncode == 0


@pytest.mark.mpi
@pytest.mark.parametrize("path,name", demos)
def test_demos_mpi(num_proc, mpiexec, path, name):
    cmd = [mpiexec, "-np", str(num_proc), sys.executable, name]
    print(cmd)
    ret = subprocess.run(cmd, cwd=str(path), check=True)
    assert ret.returncode == 0

# New test cases for setting traps, materials, and exports in festim module

def run_simulation_test(simulation, combination_type, valid_types, error_message):
    for combination in valid_types:
        setattr(simulation, combination_type, combination)

    invalid_types = ["coucou", True]

    for combination in invalid_types:
        with pytest.raises(TypeError, match=error_message):
            setattr(simulation, combination_type, combination)

def test_setting_traps():
    my_sim = F.Simulation()
    my_mat = F.Material(1, 1, 0)
    trap1 = F.Trap(1, 1, 1, 1, [my_mat], density=1)
    trap2 = F.Trap(2, 2, 2, 2, [my_mat], density=1)

    valid_types = [trap1, [trap1], [trap1, trap2], F.Traps([trap1, trap2])]
    error_message = "Accepted types for traps are list, festim.Traps or festim.Trap"

    run_simulation_test(my_sim, "traps", valid_types, error_message)

def test_setting_traps_wrong_type():
    my_sim = F.Simulation()

    invalid_types = ["coucou", True]

    for combination in invalid_types:
        with pytest.raises(
            TypeError,
            match="Accepted types for traps are list, festim.Traps or festim.Trap",
        ):
            my_sim.traps = combination

def test_setting_materials():
    my_sim = F.Simulation()
    mat1 = F.Material(1, 1, 1)
    mat2 = F.Material(2, 2, 2)

    valid_types = [mat1, [mat1], [mat1, mat2], F.Materials([mat1, mat2])]
    error_message = "accepted types for materials are list, festim.Material or festim.Materials"

    run_simulation_test(my_sim, "materials", valid_types, error_message)

def test_setting_materials_wrong_type():
    my_sim = F.Simulation()

    invalid_types = ["coucou", True]

    for combination in invalid_types:
        with pytest.raises(
            TypeError,
            match="accepted types for materials are list, festim.Material or festim.Materials",
        ):
            my_sim.materials = combination

def test_setting_exports():
    my_sim = F.Simulation()
    export1 = F.XDMFExport("solute")
    export2 = F.XDMFExport("solute")

    valid_types = [export1, [export1], [export1, export2], F.Exports([export1, export2])]
    error_message = "accepted types for exports are list, festim.Export or festim.Exports"

    run_simulation_test(my_sim, "exports", valid_types, error_message)

def test_setting_exports_wrong_type():
    my_sim = F.Simulation()

    invalid_types = ["coucou", True]

    for combination in invalid_types:
        with pytest.raises(
            TypeError,
            match="accepted types for exports are list, festim.Export or festim.Exports",
        ):
            my_sim.exports = combination
