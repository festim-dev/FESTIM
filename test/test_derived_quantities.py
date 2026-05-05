import os

from mpi4py import MPI

import numpy as np
import pytest
import ufl

import festim as F

mobile_H = F.Species("H")
mobile_D = F.Species("D")
surf_1 = F.SurfaceSubdomain(id=1)
surf_2 = F.SurfaceSubdomain(id=2)
vol_1 = F.VolumeSubdomain(id=1, material="test")
vol_2 = F.VolumeSubdomain(id=2, material="test")
results = "test.csv"

surface_flux = F.SurfaceFlux(field=mobile_H, surface=surf_1, filename=results)
average_vol = F.AverageVolume(mobile_H, volume=vol_1, filename=results)
tot_surf = F.TotalSurface(mobile_D, surface=surf_2, filename=results)
tot_vol = F.TotalVolume(mobile_D, volume=vol_2, filename=results)
min_vol = F.MinimumVolume(mobile_H, volume=vol_1, filename=results)
max_vol = F.MaximumVolume(mobile_D, volume=vol_1, filename=results)
min_surface = F.MinimumSurface(mobile_D, surface=surf_1, filename=results)
max_surface = F.MaximumSurface(mobile_H, surface=surf_2, filename=results)
avg_surface = F.AverageSurface(mobile_D, surface=surf_1, filename=results)
avg_vol = F.AverageVolume(mobile_H, volume=vol_2, filename=results)


@pytest.mark.parametrize(
    "quantity, expected_title",
    [
        (surface_flux, "H flux surface 1"),
        (average_vol, "Average H volume 1"),
        (tot_surf, "Total D surface 2"),
        (tot_vol, "Total D volume 2"),
        (min_vol, "Minimum H volume 1"),
        (max_vol, "Maximum D volume 1"),
        (min_surface, "Minimum D surface 1"),
        (max_surface, "Maximum H surface 2"),
        (avg_surface, "Average D surface 1"),
        (avg_vol, "Average H volume 2"),
    ],
)
def test_title(quantity, expected_title, tmp_path):
    quantity.filename = os.path.join(tmp_path, "test.csv")
    quantity.value = 1

    assert quantity.title == expected_title


class TestCustomQuantity:
    """Test suite for CustomQuantity export"""

    def test_custom_quantity_title_volume(self):
        """Test that CustomQuantity has correct title for volume subdomain"""

        def expr(**kwargs):
            return kwargs.get("A", 0)

        quantity = F.CustomQuantity(
            expr=expr, subdomain=vol_1, title="My Custom Quantity"
        )
        assert quantity.title == "My Custom Quantity"

    def test_custom_quantity_title_surface(self):
        """Test that CustomQuantity has correct title for surface subdomain"""

        def expr(**kwargs):
            return kwargs.get("A", 0)

        quantity = F.CustomQuantity(expr=expr, subdomain=surf_1, title="Surface Custom")
        assert quantity.title == "Surface Custom"

    def test_custom_quantity_subdomain_volume(self):
        """Test that CustomQuantity stores volume subdomain correctly"""

        def expr(**kwargs):
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1)
        assert quantity.subdomain == vol_1
        assert quantity.subdomain.id == 1

    def test_custom_quantity_subdomain_surface(self):
        """Test that CustomQuantity stores surface subdomain correctly"""

        def expr(**kwargs):
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=surf_2)
        assert quantity.subdomain == surf_2
        assert quantity.subdomain.id == 2

    def test_custom_quantity_accepts_kwargs(self):
        """Test that expression callable receives kwargs"""
        received_kwargs = {}

        def expr(**kwargs):
            received_kwargs.update(kwargs)
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1)
        quantity.expr(**{"A": 1, "B": 2, "n": None})
        assert received_kwargs == {"A": 1, "B": 2, "n": None}

    def test_custom_quantity_data_append(self):
        """Test that values are appended to data list"""

        def expr(**kwargs):
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1)
        assert quantity.data == []
        quantity.value = 5
        quantity.data.append(quantity.value)
        assert quantity.data == [5]
        quantity.value = 10
        quantity.data.append(quantity.value)
        assert quantity.data == [5, 10]

    def test_custom_quantity_t_list(self):
        """Test that time values are stored in t list"""

        def expr(**kwargs):
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1)
        assert quantity.t == []
        quantity.t.append(0.0)
        quantity.t.append(0.1)
        assert quantity.t == [0.0, 0.1]

    def test_custom_quantity_filename(self, tmp_path):
        """Test that filename can be set and retrieved"""

        def expr(**kwargs):
            return 1

        filename = os.path.join(tmp_path, "custom.csv")
        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1, filename=filename)
        assert quantity.filename == filename

    def test_custom_quantity_expr_stored(self):
        """Test that expression callable is stored correctly"""

        def my_expr(**kwargs):
            return kwargs.get("value", 0)

        quantity = F.CustomQuantity(expr=my_expr, subdomain=vol_1)
        assert quantity.expr == my_expr
        assert quantity.expr(value=42) == 42


class TestCustomQuantityWithHydrogenTransportProblem:
    """Integration tests for CustomQuantity with HydrogenTransportProblem"""

    def test_custom_quantity_volume_integration(self):
        """Test CustomQuantity with volume subdomain in HydrogenTransportProblem"""
        # BUILD
        material = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
        vol = F.VolumeSubdomain(id=1, material=material)
        top = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
        bottom = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0))

        from dolfinx.mesh import create_unit_square

        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)

        my_model = F.HydrogenTransportProblem()
        my_model.mesh = F.Mesh(mesh)
        my_model.subdomains = [vol, top, bottom]
        my_model.temperature = 300
        my_model.settings = F.Settings(
            final_time=0.02, atol=1e-6, rtol=1e-6, stepsize=0.01
        )

        species = F.Species("A")
        my_model.species = [species]

        my_model.boundary_conditions = [
            F.FixedConcentrationBC(species=species, subdomain=top, value=1),
            F.FixedConcentrationBC(species=species, subdomain=bottom, value=0),
        ]

        # Custom quantity that computes the total concentration
        def total_conc(**kwargs):
            return kwargs["A"]

        custom_qty = F.CustomQuantity(
            expr=total_conc, subdomain=vol, title="Total concentration"
        )

        my_model.exports = [custom_qty]

        # RUN
        my_model.initialise()
        my_model.run()

        # TEST
        assert len(custom_qty.data) > 0
        assert len(custom_qty.t) > 0
        assert len(custom_qty.t) == len(custom_qty.data)
        # Check that values are non-negative (concentration)
        assert all(v >= 0 for v in custom_qty.data)

    def test_custom_quantity_surface_integration(self):
        """Test CustomQuantity with surface subdomain in HydrogenTransportProblem"""
        # BUILD
        material = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
        vol = F.VolumeSubdomain(id=1, material=material)
        top = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
        bottom = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0))

        from dolfinx.mesh import create_unit_square

        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)

        my_model = F.HydrogenTransportProblem()
        my_model.mesh = F.Mesh(mesh)
        my_model.subdomains = [vol, top, bottom]
        my_model.temperature = 300
        my_model.settings = F.Settings(
            final_time=0.02, atol=1e-6, rtol=1e-6, stepsize=0.01
        )

        species = F.Species("A")
        my_model.species = [species]

        my_model.boundary_conditions = [
            F.FixedConcentrationBC(species=species, subdomain=top, value=1),
            F.FixedConcentrationBC(species=species, subdomain=bottom, value=0),
        ]

        # Custom quantity on surface
        def surface_flux(**kwargs):
            D = kwargs["D_A"]
            c = kwargs["A"]
            n = kwargs["n"]
            return -D * ufl.dot(ufl.grad(c), n)

        custom_qty = F.CustomQuantity(
            expr=surface_flux, subdomain=top, title="Surface flux"
        )

        my_model.exports = [custom_qty]

        # RUN
        my_model.initialise()
        my_model.run()

        # TEST
        assert len(custom_qty.data) > 0
        assert len(custom_qty.t) > 0
        assert len(custom_qty.t) == len(custom_qty.data)

    def test_custom_quantity_with_multiple_species(self):
        """Test CustomQuantity with multiple species"""
        # BUILD
        material = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
        vol = F.VolumeSubdomain(id=1, material=material)
        top = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
        bottom = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0))

        from dolfinx.mesh import create_unit_square

        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)

        my_model = F.HydrogenTransportProblem()
        my_model.mesh = F.Mesh(mesh)
        my_model.subdomains = [vol, top, bottom]
        my_model.temperature = 300
        my_model.settings = F.Settings(
            final_time=0.02, atol=1e-6, rtol=1e-6, stepsize=0.01
        )

        speciesA = F.Species("A")
        speciesB = F.Species("B")
        my_model.species = [speciesA, speciesB]

        my_model.boundary_conditions = [
            F.FixedConcentrationBC(species=speciesA, subdomain=top, value=1),
            F.FixedConcentrationBC(species=speciesA, subdomain=bottom, value=0),
            F.FixedConcentrationBC(species=speciesB, subdomain=top, value=0),
            F.FixedConcentrationBC(species=speciesB, subdomain=bottom, value=1),
        ]

        # Custom quantity combining both species
        def combined_conc(**kwargs):
            return kwargs["A"] + kwargs["B"]

        custom_qty = F.CustomQuantity(
            expr=combined_conc, subdomain=vol, title="Combined concentration"
        )

        my_model.exports = [custom_qty]

        # RUN
        my_model.initialise()
        my_model.run()

        # TEST
        assert len(custom_qty.data) > 0
        assert len(custom_qty.t) > 0
        # Combined concentration should be <= 2 (max of each species)
        assert all(0 <= v <= 2 for v in custom_qty.data)


class TestDerivedQuantityFilename:
    """Test suite for DerivedQuantity filename property validation"""

    def test_filename_none(self):
        """Test that filename can be set to None"""

        def expr(**kwargs):
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1, filename=None)
        assert quantity.filename is None

    def test_filename_csv(self, tmp_path):
        """Test that filename accepts .csv extension"""

        def expr(**kwargs):
            return 1

        filename = os.path.join(tmp_path, "output.csv")
        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1, filename=filename)
        assert quantity.filename == filename

    def test_filename_txt(self, tmp_path):
        """Test that filename accepts .txt extension"""

        def expr(**kwargs):
            return 1

        filename = os.path.join(tmp_path, "output.txt")
        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1, filename=filename)
        assert quantity.filename == filename

    def test_filename_invalid_extension(self):
        """Test that filename rejects invalid extensions"""

        def expr(**kwargs):
            return 1

        with pytest.raises(ValueError, match="filename must end with .csv or .txt"):  # noqa: RUF043
            F.CustomQuantity(expr=expr, subdomain=vol_1, filename="output.dat")

    def test_filename_invalid_type(self):
        """Test that filename rejects non-string types"""

        def expr(**kwargs):
            return 1

        with pytest.raises(TypeError, match="filename must be of type str"):
            F.CustomQuantity(expr=expr, subdomain=vol_1, filename=123)

    def test_filename_property_setter_none(self):
        """Test setting filename property to None after initialization"""

        def expr(**kwargs):
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1)
        quantity.filename = None
        assert quantity.filename is None

    def test_filename_property_setter_csv(self, tmp_path):
        """Test setting filename property to a .csv file"""

        def expr(**kwargs):
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1)
        filename = os.path.join(tmp_path, "new_output.csv")
        quantity.filename = filename
        assert quantity.filename == filename

    def test_filename_property_setter_txt(self, tmp_path):
        """Test setting filename property to a .txt file"""

        def expr(**kwargs):
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1)
        filename = os.path.join(tmp_path, "new_output.txt")
        quantity.filename = filename
        assert quantity.filename == filename

    def test_filename_property_setter_invalid_extension(self):
        """Test that setting invalid extension raises ValueError"""

        def expr(**kwargs):
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1)
        with pytest.raises(ValueError, match="filename must end with .csv or .txt"):  # noqa: RUF043
            quantity.filename = "output.json"

    def test_filename_property_setter_invalid_type(self):
        """Test that setting non-string filename raises TypeError"""

        def expr(**kwargs):
            return 1

        quantity = F.CustomQuantity(expr=expr, subdomain=vol_1)
        with pytest.raises(TypeError, match="filename must be of type str"):
            quantity.filename = 42


class TestCustomQuantityWithDiscontinuousProblem:
    """
    Integration tests for CustomQuantity with HydrogenTransportProblemDiscontinuous
    """

    def test_custom_quantity_discontinuous_volume(self):
        """Test CustomQuantity with volume subdomain in discontinuous problem"""
        # BUILD
        material = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
        vol = F.VolumeSubdomain(id=1, material=material)
        top = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
        bottom = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0))

        from dolfinx.mesh import create_unit_square

        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)

        my_model = F.HydrogenTransportProblemDiscontinuous()
        my_model.mesh = F.Mesh(mesh)
        my_model.subdomains = [vol, top, bottom]
        my_model.temperature = 300
        my_model.settings = F.Settings(
            final_time=0.02, atol=1e-6, rtol=1e-6, stepsize=0.01
        )

        species = F.Species("A", subdomains=[vol])
        my_model.species = [species]

        my_model.boundary_conditions = [
            F.FixedConcentrationBC(species=species, subdomain=top, value=1),
            F.FixedConcentrationBC(species=species, subdomain=bottom, value=0),
        ]

        # Custom quantity
        def total_conc(**kwargs):
            return kwargs["A"]

        custom_qty = F.CustomQuantity(
            expr=total_conc, subdomain=vol, title="Total concentration"
        )

        my_model.exports = [custom_qty]

        # RUN
        my_model.initialise()
        my_model.run()

        # TEST
        assert len(custom_qty.data) > 0
        assert len(custom_qty.t) > 0
        assert len(custom_qty.t) == len(custom_qty.data)

    def test_custom_quantity_discontinuous_surface(self):
        """Test CustomQuantity with surface subdomain in discontinuous problem"""
        # BUILD
        material = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)
        vol = F.VolumeSubdomain(id=1, material=material)
        top = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1))
        bottom = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0))

        from dolfinx.mesh import create_unit_square

        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)

        my_model = F.HydrogenTransportProblemDiscontinuous()
        my_model.mesh = F.Mesh(mesh)
        my_model.subdomains = [vol, top, bottom]
        my_model.temperature = 300
        my_model.settings = F.Settings(
            final_time=0.02, atol=1e-6, rtol=1e-6, stepsize=0.01
        )

        species = F.Species("A", subdomains=[vol])
        my_model.species = [species]

        my_model.boundary_conditions = [
            F.FixedConcentrationBC(species=species, subdomain=top, value=1),
            F.FixedConcentrationBC(species=species, subdomain=bottom, value=0),
        ]

        # Custom quantity on surface
        def surface_flux(**kwargs):
            D = kwargs["D_A"]
            c = kwargs["A"]
            n = kwargs["n"]
            return -D * ufl.dot(ufl.grad(c), n)

        custom_qty = F.CustomQuantity(
            expr=surface_flux, subdomain=top, title="Surface flux"
        )

        my_model.exports = [custom_qty]

        # RUN
        my_model.initialise()
        my_model.run()

        # TEST
        assert len(custom_qty.data) > 0
        assert len(custom_qty.t) > 0
        assert len(custom_qty.t) == len(custom_qty.data)
