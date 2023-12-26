from lys_fem.fem import DomainCondition

class Source(DomainCondition):
    className = "Source Term"

    @classmethod
    def default(cls, model):
        return Source([0]*model.variableDimension())

    def widget(self, fem, canvas):
        pass
