import numpy as np
from lys_fem.fem import DomainCondition, Coef

class Source(DomainCondition):
    className = "Source Term"
    def __init__(self, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["value"] = Coef(value, shape=np.shape(value), description = "Source")

    @classmethod
    def default(cls, fem, model):
        return Source([0]*model.variableDimension)


class DivSource(DomainCondition):
    className = "Div Source"
    def __init__(self, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["value"] = Coef(value, shape=np.shape(value), description = "Div source")

    @classmethod
    def default(cls, fem, model):
        return DivSource([0]*model.variableDimension)