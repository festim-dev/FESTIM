class Material:
    """
    Args:
        id (int, list): the id of the material. If a list is provided, the
            properties will be applied to all the subdomains with the
            corresponding ids.
        D_0 (float): diffusion coefficient pre-exponential factor (m2/s)
        E_D (float): diffusion coefficient activation energy (eV)
        S_0 (float, optional): Solubility pre-exponential factor
            (H/m3/Pa0.5). Defaults to None.
        E_S (float, optional): Solubility activation energy (eV).
            Defaults to None.
        thermal_cond (float or callable, optional): thermal conductivity
            (W/m/K). Can be a function of T. Defaults to None.
        heat_capacity (float or callable, optional): heat capacity
            (J/K/kg). Can be a function of T. Defaults to None.
        rho (float or callable, optional): volumetric density (kg/m3). Can
            be a function of T. Defaults to None.
        borders (list, optional): The borders of the 1D subdomain.
            Only needed in 1D with several materials. Defaults to None.
        H (dict, optional): heat of transport (J/mol).
            {"free_enthalpy": ..., "entropy": ...} so that
            H = free_enthalpy + entropy*T. Defaults to None.
        solubility_law (str, optional): the material's solubility law.
            Can be "henry" or "sievert". Defaults to "sievert".
        name (str, optional): name of the material. Defaults to None.
    """

    def __init__(
        self,
        id,
        D_0,
        E_D,
        S_0=None,
        E_S=None,
        thermal_cond=None,
        heat_capacity=None,
        rho=None,
        borders=None,
        H=None,
        solubility_law="sievert",
        name=None,
    ) -> None:

        self.id = id
        self.name = name
        self.D_0 = D_0
        self.E_D = E_D
        self.S_0 = S_0
        self.E_S = E_S
        self.thermal_cond = thermal_cond
        self.heat_capacity = heat_capacity
        self.rho = rho
        self.borders = borders
        self.H = H
        if H is not None:
            self.free_enthalpy = H["free_enthalpy"]
            self.entropy = H["entropy"]
        if solubility_law not in ["henry", "sievert"]:
            raise ValueError(
                "Acceptable values for solubility_law are 'henry' and 'sievert'"
            )
        self.solubility_law = solubility_law
        self.check_properties()

    def check_properties(self):
        """Checks that if S_0 is None E_S is not None and reverse.

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        if self.S_0 is None and self.E_S is not None:
            raise ValueError("S_0 cannot be None")
        if self.E_S is None and self.S_0 is not None:
            raise ValueError("E_S cannot be None")
