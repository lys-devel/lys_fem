from lys_fem import FEMParameter


class SemiconductorParameters(FEMParameter):
    name = "Semiconductor Drift Diffusion"

    def __init__(self, mu_n=0.15, mu_p=0.15, N_d=0.0, N_a=0.0):
        self.mu_n = mu_n
        self.mu_p = mu_p
        self.N_d = N_d
        self.N_a = N_a

    def getParameters(self):
        res = {}
        if self.mu_n is not None:
            res["mu_n"] = self.mu_n
        if self.mu_p is not None:
            res["mu_p"] = self.mu_p
        if self.N_d is not None:
            res["N_d"] = self.N_d
        if self.N_a is not None:
            res["N_a"] = self.N_a
        return res

    @property
    def description(self):
        return {
            "mu_n": "Electron mobility (m^2/V s)",
            "mu_p": "Hole mobility (m^2/V s)",
            "N_d": "Donor density (1/m^3)",
            "N_a": "Acceptor density (1/m^3)",
        }

    @property
    def default(self):
        return {
            "mu_n": 0.15,
            "mu_p": 0.15,
            "N_d": 0,
            "N_a": 0,
        }


