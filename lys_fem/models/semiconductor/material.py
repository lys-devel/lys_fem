from lys_fem import FEMParameter, Coef


class SemiconductorParameters(FEMParameter):
    name = "Semiconductor Drift Diffusion"

    def __init__(self, mu_n=0.15, mu_p=0.15, N_d=0.0, N_a=0.0):
        self["mu_n"] = Coef(mu_n, description="Electron mobility (m^2/V s)")
        self["mu_p"] = Coef(mu_p, description="Hole mobility (m^2/V s)")
        self["N_d"] = Coef(N_d, description="Donor density (1/m^3)")
        self["N_a"] = Coef(N_a, description="Acceptor density (1/m^3)")

