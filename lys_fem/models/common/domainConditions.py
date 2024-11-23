from lys_fem.fem import DomainCondition

class Source(DomainCondition):
    className = "Source Term"

    @classmethod
    def default(cls, fem, model):
        return Source([0]*model.variableDimension())


class DivSource(DomainCondition):
    className = "Div Source"

    @classmethod
    def default(cls, fem, model):
        return DivSource([0]*model.variableDimension())