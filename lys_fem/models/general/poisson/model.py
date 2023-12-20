from lys_fem import FEMFixedModel, DomainCondition, GeometrySelection


class PoissonModel(FEMFixedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    @classmethod
    @property
    def name(cls):
        return "Poisson"

    @property
    def variableName(self):
        return "phi"

    def evalList(self):
        return ["phi"]

    def eval(self, data, fem, var):
        if var == "phi":
            return data["phi"]


class InfiniteVolume(DomainCondition):
    def __init__(self, name, domains=None):
        super().__init__(name, domains)

    @classmethod
    @property
    def name(cls):
        return "Infinite Volume"

    def widget(self, fem, canvas):
        pass

    def saveAsDictionary(self):
        return {"type": self.name, "name": self.objName, "values": self._value, "domains": self.domains.saveAsDictionary()}

    @staticmethod
    def loadFromDictionary(d):
        domains = GeometrySelection.loadFromDictionary(d["domains"])
        return InfiniteVolume(d["name"], domains=domains)

