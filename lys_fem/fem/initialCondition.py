from .geometry import GeometrySelection


class InitialCondition:
    def __init__(self, name, value, domains=None):
        if isinstance(domains, GeometrySelection):
            self._domains = domains
        else:
            self._domains = GeometrySelection("Domain", domains)
        self._name = name
        self._value = value

    def setDimension(self, dim):
        if len(self._value) > dim:
            self._value = self._value[:2]
        while len(self._value) < dim:
            self._value.append("0")

    @classmethod
    @property
    def name(cls):
        return "Initial Condition"

    @property
    def objName(self):
        return self._name

    @property
    def domains(self):
        return self._domains

    @property
    def values(self):
        return self._value

    @values.setter
    def values(self, value):
        self._value = value

    def widget(self, fem, canvas):
        from ..gui import InitialConditionWidget
        return InitialConditionWidget(self, fem, canvas)

    def saveAsDictionary(self):
        return {"type": self.name, "name": self.objName, "domains": self._domains.saveAsDictionary(), "values": self._value}

    @staticmethod
    def loadFromDictionary(d):
        values = d["values"]
        domain = GeometrySelection.loadFromDictionary(d["domains"])
        return InitialCondition(d["name"], value=values, domains=domain)
