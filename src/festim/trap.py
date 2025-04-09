from festim.reaction import Reaction as _Reaction
from festim.species import ImplicitSpecies as _ImplicitSpecies
from festim.species import Species as _Species


class Trap(_Species):
    """Trap species class for H transport simulation.

    This class only works for 1 mobile species and 1 trapping level and is
    for convenience, for more details see notes.

    Args:
        name (str, optional): a name given to the trap. Defaults to None.
        mobile_species (_Species): the mobile species to be trapped
        k_0 (float): the trapping rate constant pre-exponential factor (m3 s-1)
        E_k (float): the trapping rate constant activation energy (eV)
        p_0 (float): the detrapping rate constant pre-exponential factor (s-1)
        E_p (float): the detrapping rate constant activation energy (eV)
        volume (F.VolumeSubdomain1D): The volume subdomain where the trap is.

    Attributes:
        name (str, optional): a name given to the trap. Defaults to None.
        mobile_species (_Species): the mobile species to be trapped
        k_0 (float): the trapping rate constant pre-exponential factor (m3 s-1)
        E_k (float): the trapping rate constant activation energy (eV)
        p_0 (float): the detrapping rate constant pre-exponential factor (s-1)
        E_p (float): the detrapping rate constant activation energy (eV)
        volume (F.VolumeSubdomain1D): The volume subdomain where the trap is.
        trapped_concentration (_Species): The immobile trapped concentration
        trap_reaction (_Reaction): The reaction for trapping the mobile conc.
        empty_trap_sites (F.ImplicitSpecies): The implicit species for the empty trap sites

    Examples:

        .. testsetup:: Trap

            from festim import Trap, Species, VolumeSubdomain, Material, HydrogenTransportProblem

            my_mat = Material(D_0=1, E_D=1, name="test_mat")
            my_vol = VolumeSubdomain(id=1, material=my_mat)
            H = Species(name="H")

        .. testcode:: Trap

            trap = Trap(name="Trap", mobile_species=H, k_0=1.0, E_k=0.2, p_0=0.1, E_p=0.3, n=100, volume=my_vol)

            my_model = HydrogenTransportProblem()
            my_model.traps = [trap]

    Notes:
        This convenience class replaces the need to specify an implicit species and
        the associated reaction, thus:

        .. code:: python

            cm = _Species("mobile")
            my_trap = F.Trap(
                name="trapped",
                mobile_species=cm,
                k_0=1,
                E_k=1,
                p_0=1,
                E_p=1,
                n=1,
                volume=my_vol,
            )
            my_model.species = [cm]
            my_model.traps = [my_trap]

        is equivalent to:

        .. code:: python

            cm = _Species("mobile")
            ct = _Species("trapped")
            trap_sites = F.ImplicitSpecies(n=1, others=[ct])
            trap_reaction = _Reaction(
                reactant=[cm, trap_sites],
                product=ct,
                k_0=1,
                E_k=1,
                p_0=1,
                E_p=1,
                volume=my_vol,
            )
            my_model.species = [cm, ct]
            my_model.reactions = [trap_reaction]


    """

    def __init__(
        self, name: str, mobile_species, k_0, E_k, p_0, E_p, n, volume
    ) -> None:
        super().__init__(name)
        self.mobile_species = mobile_species
        self.k_0 = k_0
        self.E_k = E_k
        self.p_0 = p_0
        self.E_p = E_p
        self.n = n
        self.volume = volume

        self.trapped_concentration = None
        self.reaction = None

    def create_species_and_reaction(self):
        """create the immobile trapped species object and the reaction for trapping"""
        self.trapped_concentration = _Species(name=self.name, mobile=False)
        self.empty_trap_sites = _ImplicitSpecies(
            n=self.n, others=[self.trapped_concentration]
        )
        self.reaction = _Reaction(
            reactant=[self.mobile_species, self.empty_trap_sites],
            product=self.trapped_concentration,
            k_0=self.k_0,
            E_k=self.E_k,
            p_0=self.p_0,
            E_p=self.E_p,
            volume=self.volume,
        )
