from lys_fem import FEMFixedModel, Equation


class HeatConductionModel(FEMFixedModel):
    className = "Heat Conduction"

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    @classmethod
    @property
    def equationTypes(self):
        return [HeatConductionEquation]
    

class HeatConductionEquation(Equation):
    className = "Heat Conduction Equation"
    def __init__(self, varName="T", **kwargs):
        super().__init__(varName, **kwargs)
