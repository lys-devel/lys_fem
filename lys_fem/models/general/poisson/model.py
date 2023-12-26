from lys_fem import FEMFixedModel, Equation
from lys_fem.models.common import Source

class PoissonModel(FEMFixedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1, [PoissonEquation("phi")], *args, **kwargs)

    @classmethod
    @property
    def name(cls):
        return "Poisson"

    @classmethod
    @property
    def domainConditionTypes(cls):
        return [Source]

    @property
    def variableName(self):
        return "phi"

    def evalList(self):
        return ["phi"]

    def eval(self, data, fem, var):
        if var == "phi":
            return data["phi"]


class PoissonEquation(Equation):
    def __init__(self, varName, domain="all", name="Poisson Equation"):
        super().__init__(name, varName, geometries = domain)



