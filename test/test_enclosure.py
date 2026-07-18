import re

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import festim as F

from .markers import requires_dolfinx_010, requires_dolfinx_011


def make_gas_species(**kwargs):
    return F.GasSpecies(name=kwargs.pop("name", "H2"), **kwargs)


def make_enclosure(**kwargs):
    kwargs.setdefault("volume", 1e-3)
    kwargs.setdefault("species", [make_gas_species()])
    kwargs.setdefault("temperature", 500.0)
    return F.Enclosure(**kwargs)


def make_model(enclosures=None, transient=True):
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 8)
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh=mesh)
    material = F.Material(name="mat", D_0=1e-9, E_D=0.0)
    volume = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=material)
    left = F.SurfaceSubdomain1D(id=1, x=0.0)
    right = F.SurfaceSubdomain1D(id=2, x=1.0)
    my_model.subdomains = [volume, left, right]
    my_model.species = [F.Species("H", subdomains=[volume])]
    my_model.temperature = 500.0
    my_model.enclosures = enclosures or []
    my_model.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=transient,
        final_time=1.0,
        stepsize=F.Stepsize(1.0) if transient else None,
    )
    my_model.show_progress_bar = False
    return my_model, volume, left, right


class TestConstruction:
    def test_backref_is_set(self):
        H2 = make_gas_species()
        enclosure = make_enclosure(species=[H2])
        assert H2.enclosure is enclosure

    def test_negative_volume_raises(self):
        with pytest.raises(ValueError, match="volume must be positive"):
            make_enclosure(volume=-1.0)

    def test_empty_species_raises(self):
        with pytest.raises(ValueError, match="at least one GasSpecies"):
            make_enclosure(species=[])

    def test_wrong_species_type_raises(self):
        with pytest.raises(TypeError, match="list of GasSpecies"):
            make_enclosure(species=[F.Species("H")])

    def test_gas_constant_defaults_to_boltzmann_si(self):
        assert make_enclosure().gas_constant == F.k_B_SI

    def test_gas_constant_can_be_molar(self):
        assert make_enclosure(gas_constant=F.R).gas_constant == F.R

    def test_pressure_unavailable_before_initialise(self):
        with pytest.raises(ValueError, match="not available before initialise"):
            _ = make_gas_species().value


class TestEnclosureConnection:
    def test_needs_exactly_two_species(self):
        H2 = make_gas_species()
        with pytest.raises(ValueError, match="exactly two gas species"):
            F.EnclosureConnection(conductance=1e-4, species=(H2,))

    def test_partner_lookup(self):
        a, b = make_gas_species(name="a"), make_gas_species(name="b")
        connection = F.EnclosureConnection(conductance=1e-4, species=(a, b))
        assert connection._partner(a) is b
        assert connection._partner(b) is a

    def test_partner_lookup_rejects_unrelated_species(self):
        a, b = make_gas_species(name="a"), make_gas_species(name="b")
        connection = F.EnclosureConnection(conductance=1e-4, species=(a, b))
        with pytest.raises(ValueError, match="not connected by"):
            connection._partner(make_gas_species(name="c"))

    @requires_dolfinx_011
    def test_species_must_belong_to_an_enclosure(self):
        a, b = make_gas_species(name="a"), make_gas_species(name="b")
        connection = F.EnclosureConnection(conductance=1e-4, species=(a, b))
        # b is never put in an enclosure
        enclosure_a = make_enclosure(species=[a], openings=[connection])
        my_model, *_ = make_model(enclosures=[enclosure_a])
        with pytest.raises(ValueError, match="does not belong to any enclosure"):
            my_model.initialise()

    @requires_dolfinx_011
    def test_connection_is_mirrored_onto_partner(self):
        a, b = make_gas_species(name="a"), make_gas_species(name="b")
        connection = F.EnclosureConnection(conductance=1e-4, species=(a, b))
        enclosure_a = make_enclosure(species=[a], openings=[connection])
        enclosure_b = make_enclosure(species=[b])
        assert connection not in enclosure_b.openings

        my_model, *_ = make_model(enclosures=[enclosure_a, enclosure_b])
        my_model.initialise()
        assert connection in enclosure_b.openings

    @requires_dolfinx_011
    def test_mirroring_is_idempotent(self):
        """Declaring the connection on both sides must not double the flow."""
        a, b = make_gas_species(name="a"), make_gas_species(name="b")
        connection = F.EnclosureConnection(conductance=1e-4, species=(a, b))
        enclosure_a = make_enclosure(species=[a], openings=[connection])
        enclosure_b = make_enclosure(species=[b], openings=[connection])
        my_model, *_ = make_model(enclosures=[enclosure_a, enclosure_b])
        my_model.initialise()
        assert enclosure_b.openings.count(connection) == 1


class TestOpenings:
    def test_applies_to_all_species_by_default(self):
        H2, HD = make_gas_species(name="H2"), make_gas_species(name="HD")
        pump = F.Pump(pumping_speed=1e-4)
        assert pump.applies_to(H2)
        assert pump.applies_to(HD)

    def test_applies_to_named_species_only(self):
        H2, HD = make_gas_species(name="H2"), make_gas_species(name="HD")
        pump = F.Pump(pumping_speed=1e-4, species=H2)
        assert pump.applies_to(H2)
        assert not pump.applies_to(HD)

    @pytest.mark.parametrize(
        "opening, time_dependent",
        [
            (F.Pump(pumping_speed=1e-4), False),
            (F.Pump(pumping_speed=lambda t: 1e-4 * t), True),
            (F.Reservoir(conductance=1e-4, pressure=1e3), False),
            (F.Reservoir(conductance=1e-4, pressure=lambda t: 1e3 * t), True),
            (F.PrescribedFlowRate(flow_rate=1e18), False),
            (F.PrescribedFlowRate(flow_rate=lambda t: 1e18 * t), True),
        ],
    )
    def test_time_dependence_detection(self, opening, time_dependent):
        assert any(v.explicit_time_dependent for v in opening._values) == time_dependent


@requires_dolfinx_011
class TestOpeningSigns:
    """The sign convention: molar_flow_rate is positive when particles enter."""

    def _initialised(self, opening):
        H2 = make_gas_species(initial_pressure=1e5)
        enclosure = make_enclosure(species=[H2], openings=[opening])
        my_model, *_ = make_model(enclosures=[enclosure])
        my_model.initialise()
        return H2, enclosure, my_model

    def _assemble(self, expr, my_model):
        return dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(
                expr
                / my_model._total_volume
                * dolfinx.fem.Constant(my_model.mesh.mesh, 1.0)
                * my_model.dx
            )
        )

    def test_pump_removes_particles(self):
        H2, enclosure, my_model = self._initialised(F.Pump(pumping_speed=1e-4))
        rate = self._assemble(
            enclosure.openings[0].molar_flow_rate(H2, enclosure), my_model
        )
        assert rate < 0

    def test_reservoir_at_higher_pressure_adds_particles(self):
        H2, enclosure, my_model = self._initialised(
            F.Reservoir(conductance=1e-4, pressure=1e6)  # above the 1e5 enclosure
        )
        rate = self._assemble(
            enclosure.openings[0].molar_flow_rate(H2, enclosure), my_model
        )
        assert rate > 0

    def test_reservoir_at_lower_pressure_removes_particles(self):
        H2, enclosure, my_model = self._initialised(
            F.Reservoir(conductance=1e-4, pressure=1e3)
        )
        rate = self._assemble(
            enclosure.openings[0].molar_flow_rate(H2, enclosure), my_model
        )
        assert rate < 0


@requires_dolfinx_011
class TestValidation:
    def test_area_required_on_1d_mesh(self):
        """In 1D a surface is a point, so the mesh cannot supply the area of the
        membrane facing the enclosure and the user must give it."""
        H2 = make_gas_species()
        my_model, _volume, _left, right = make_model()
        # a plain list means no areas were given
        my_model.enclosures = [make_enclosure(species=[H2], surfaces=[right])]
        with pytest.raises(ValueError, match="areas of those surfaces"):
            my_model.initialise()

    def test_area_given_as_dict_is_accepted_on_1d_mesh(self):
        H2 = make_gas_species()
        my_model, _volume, _left, right = make_model()
        my_model.enclosures = [make_enclosure(species=[H2], surfaces={right: 1e-4})]
        my_model.initialise()  # must not raise

    def test_surfaceless_enclosure_needs_no_area(self):
        """An enclosure that only has openings never integrates a flux, so there is no
        area to supply."""
        H2 = make_gas_species()
        enclosure = make_enclosure(species=[H2], openings=[F.Pump(pumping_speed=1e-4)])
        my_model, *_ = make_model(enclosures=[enclosure])
        my_model.initialise()  # must not raise

    def test_negative_area_raises(self):
        H2 = make_gas_species()
        right = F.SurfaceSubdomain1D(id=2, x=1.0)
        with pytest.raises(ValueError, match="area of surface 2 must be positive"):
            make_enclosure(species=[H2], surfaces={right: -1.0})

    def test_surfaces_keys_must_be_surface_subdomains(self):
        H2 = make_gas_species()
        with pytest.raises(TypeError, match="SurfaceSubdomain"):
            make_enclosure(species=[H2], surfaces={"right": 1.0})

    def test_surface_not_in_subdomains_raises(self):
        stray = F.SurfaceSubdomain1D(id=99, x=0.5)
        H2 = make_gas_species()
        enclosure = make_enclosure(species=[H2], surfaces={stray: 1.0})
        my_model, *_ = make_model(enclosures=[enclosure])
        with pytest.raises(ValueError, match="is not in the subdomains"):
            my_model.initialise()

    def test_same_gas_species_in_two_enclosures_raises(self):
        """A GasSpecies has a single pressure unknown, so it cannot live in two
        enclosures. Sharing one would otherwise silently corrupt the model."""
        H2 = make_gas_species()
        enclosure_a = make_enclosure(species=[H2])
        enclosure_b = make_enclosure(species=[H2])
        my_model, *_ = make_model(enclosures=[enclosure_a, enclosure_b])
        with pytest.raises(ValueError, match="belongs to more than one enclosure"):
            my_model.initialise()

    def test_steady_state_enclosure_with_nothing_coupled_raises(self):
        """Steady state with no surfaces and no openings: the pressure never appears in
        its own balance, so it is undetermined."""
        H2 = make_gas_species()
        enclosure = make_enclosure(species=[H2])
        my_model, *_ = make_model(enclosures=[enclosure], transient=False)
        with pytest.raises(ValueError, match="undetermined"):
            my_model.initialise()

    def test_steady_state_with_prescribed_flow_rate_only_raises(self):
        """A PrescribedFlowRate is an opening, but its rate does not depend on the
        pressure, so in steady state the pressure is still undetermined."""
        H2 = make_gas_species()
        enclosure = make_enclosure(
            species=[H2], openings=[F.PrescribedFlowRate(flow_rate=1e18)]
        )
        my_model, *_ = make_model(enclosures=[enclosure], transient=False)
        with pytest.raises(ValueError, match="undetermined"):
            my_model.initialise()

    @pytest.mark.parametrize(
        "opening",
        [F.Pump(pumping_speed=1e-4), F.Reservoir(conductance=1e-4, pressure=1e3)],
    )
    def test_steady_state_with_pressure_dependent_opening_is_accepted(self, opening):
        """A Pump or Reservoir rate depends on the pressure, so it determines it."""
        H2 = make_gas_species()
        enclosure = make_enclosure(species=[H2], openings=[opening])
        my_model, *_ = make_model(enclosures=[enclosure], transient=False)
        my_model.initialise()  # must not raise

    def test_transient_closed_enclosure_is_accepted(self):
        """In transient the time derivative always determines the pressure."""
        H2 = make_gas_species()
        enclosure = make_enclosure(species=[H2])
        my_model, *_ = make_model(enclosures=[enclosure], transient=True)
        my_model.initialise()  # must not raise

    def test_base_problem_rejects_enclosures(self):
        H2 = make_gas_species()
        my_model = F.HydrogenTransportProblem()
        my_model.enclosures = [make_enclosure(species=[H2])]
        with pytest.raises(NotImplementedError, match="Discontinuous"):
            my_model.initialise()

    def test_cylindrical_coordinates_raise(self):
        H2 = make_gas_species()
        my_model, *_ = make_model(enclosures=[make_enclosure(species=[H2])])
        my_model.mesh = F.Mesh1D(
            vertices=np.linspace(0, 1, 9), coordinate_system="cylindrical"
        )
        with pytest.raises(NotImplementedError, match="cartesian"):
            my_model.initialise()


@requires_dolfinx_011
class TestSurfaceReactionCoupling:
    def test_gas_pressure_accepts_gas_species(self):
        H2 = make_gas_species(initial_pressure=1e5)
        my_model, _volume, _left, right = make_model()
        enclosure = make_enclosure(species=[H2], surfaces={right: 1.0})
        my_model.enclosures = [enclosure]
        H = my_model.species[0]
        my_model.boundary_conditions = [
            F.SurfaceReactionBC(
                reactant=[H, H],
                gas_pressure=H2,
                k_r0=1e-4,
                E_kr=0.0,
                k_d0=1e-4,
                E_kd=0.0,
                subdomain=right,
            )
        ]
        my_model.initialise()
        # the reaction flux must depend on the pressure unknown
        value = my_model.boundary_conditions[0].flux_bcs[0].value_fenics
        assert H2.solution in ufl.algorithms.extract_coefficients(value)

    def test_enclosure_residual_depends_on_the_species(self):
        """The pressure balance must see the solid, otherwise the coupling is one-way
        and the gas would never gain the particles the solid loses."""
        H2 = make_gas_species(initial_pressure=1e5)
        my_model, volume, _left, right = make_model()
        my_model.enclosures = [make_enclosure(species=[H2], surfaces={right: 1.0})]
        H = my_model.species[0]
        my_model.boundary_conditions = [
            F.SurfaceReactionBC(
                reactant=[H, H],
                gas_pressure=H2,
                k_r0=1e-4,
                E_kr=0.0,
                k_d0=1e-4,
                E_kd=0.0,
                subdomain=right,
            )
        ]
        my_model.initialise()
        coefficients = ufl.algorithms.extract_coefficients(H2.F)
        assert volume.u in coefficients


class TestGasPressureExport:
    def test_field_must_be_gas_species(self):
        with pytest.raises(TypeError, match=re.escape("must be a festim.GasSpecies")):
            F.GasPressure(field=F.Species("H"))

    def test_title(self):
        H2 = make_gas_species(name="H2")
        make_enclosure(species=[H2], name="plenum")
        assert F.GasPressure(field=H2).title == "H2 pressure (plenum) (Pa)"

    def test_title_without_enclosure_name(self):
        H2 = make_gas_species(name="H2")
        make_enclosure(species=[H2])
        assert F.GasPressure(field=H2).title == "H2 pressure (Pa)"


class TestDirichletCoupling:
    """Henry's and Sieverts' laws coupled to an enclosure pressure."""

    def make_coupled_model(self, bc_class, enforce_weakly=True, penalty=100, **kwargs):
        H2 = make_gas_species(name="H2", initial_pressure=1e5)
        my_model, _, left, _ = make_model()
        enclosure = F.Enclosure(
            volume=1e-3, species=[H2], temperature=500.0, surfaces={left: 1.0}
        )
        my_model.enclosures = [enclosure]
        key = "H_0" if bc_class is F.HenrysBC else "S_0"
        energy = "E_H" if bc_class is F.HenrysBC else "E_S"
        my_model.boundary_conditions = [
            bc_class(
                subdomain=left,
                pressure=H2,
                species=my_model.species[0],
                enforce_weakly=enforce_weakly,
                penalty=penalty,
                **{key: 1e15, energy: 0.0},
                **kwargs,
            )
        ]
        return my_model, H2

    @pytest.mark.parametrize("bc_class", [F.HenrysBC, F.SievertsBC])
    def test_strong_enforcement_raises(self, bc_class):
        """The value depends on a real-space unknown, so it cannot be interpolated into
        a fem.Function. Enabling weak enforcement silently would change the
        discretisation behind the user's back, so it has to be asked for."""
        my_model, _ = self.make_coupled_model(bc_class, enforce_weakly=False)
        with pytest.raises(ValueError, match="can only be enforced weakly"):
            my_model.initialise()

    @pytest.mark.parametrize("bc_class", [F.HenrysBC, F.SievertsBC])
    def test_missing_penalty_raises(self, bc_class):
        """There is no defensible default penalty, so it must be given explicitly."""
        my_model, _ = self.make_coupled_model(bc_class, penalty=None)
        with pytest.raises(ValueError, match="can only be enforced weakly"):
            my_model.initialise()

    @pytest.mark.parametrize("bc_class, expected", [(F.HenrysBC, 1), (F.SievertsBC, 2)])
    def test_stoichiometry(self, bc_class, expected):
        """Sieverts' law dissolves a diatomic molecule as two atoms, Henry's law
        dissolves the molecule as such."""
        assert bc_class.stoichiometry == expected

    @pytest.mark.parametrize("bc_class", [F.HenrysBC, F.SievertsBC])
    def test_value_is_a_ufl_expression_of_the_pressure(self, bc_class):
        """The value must stay a ufl expression carrying the pressure unknown, rather
        than being interpolated, so that Newton sees the coupling."""
        my_model, H2 = self.make_coupled_model(bc_class)
        my_model.initialise()
        bc = my_model.boundary_conditions[0]
        assert not isinstance(bc.value_fenics, dolfinx.fem.Function)
        assert H2.solution in ufl.algorithms.extract_coefficients(bc.value_fenics)

    @pytest.mark.parametrize("bc_class", [F.HenrysBC, F.SievertsBC])
    def test_pressure_appears_in_its_own_balance(self, bc_class):
        """The flux through the coupled surface must reach the pressure balance,
        otherwise the pressure would be undetermined."""
        my_model, H2 = self.make_coupled_model(bc_class)
        my_model.initialise()
        assert H2.solution in ufl.algorithms.extract_coefficients(H2.F)

    @pytest.mark.parametrize("bc_class", [F.HenrysBC, F.SievertsBC])
    def test_solid_solution_appears_in_the_pressure_balance(self, bc_class):
        """The pressure balance is driven by the numerical flux, which depends on the
        concentration in the solid."""
        my_model, H2 = self.make_coupled_model(bc_class)
        my_model.initialise()
        volume = my_model.volume_subdomains[0]
        assert volume.u in ufl.algorithms.extract_coefficients(H2.F)

    def test_space_or_time_dependent_value_raises(self):
        """A value that also depends on x or t would need interpolating, which is
        exactly what a real-space coefficient forbids."""
        bc = F.HenrysBC(
            subdomain=F.SurfaceSubdomain(id=1),
            H_0=1e15,
            E_H=0.0,
            pressure=lambda t: 1e5 + t,
            species=F.Species("H"),
        )
        with pytest.raises(ValueError, match="cannot also"):
            bc.create_value_ufl(temperature=500.0)


@requires_dolfinx_010
def test_enclosures_rejected_on_old_dolfinx():
    """On dolfinx < 0.11 the feature must fail early with a helpful message."""
    H2 = F.GasSpecies(name="H2")
    enclosure = F.Enclosure(volume=1e-3, species=[H2], temperature=500.0)
    my_model = F.HydrogenTransportProblemDiscontinuous()
    with pytest.raises(NotImplementedError, match=re.escape("require dolfinx >= 0.11")):
        my_model.enclosures = [enclosure]
