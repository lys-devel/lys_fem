from .geometry import GeometrySelection


class DomainCondition:
    def __init__(self, name, domains=None):
        self._name = name
        if isinstance(domains, GeometrySelection):
            self._domains = domains
        else:
            self._domains = GeometrySelection("Domain", domains)

    @property
    def objName(self):
        return self._name

    @property
    def domains(self):
        return self._domains

    @domains.setter
    def domains(self, value):
        self._domains = value


class Source(DomainCondition):
    def __init__(self, name, values, domains=None):
        super().__init__(name, domains)
        self._value = [str(v) for v in values]

    @classmethod
    @property
    def name(cls):
        return "Source Term"

    def widget(self, fem, canvas):
        pass

    def saveAsDictionary(self):
        return {"type": self.name, "name": self.objName, "values": self._value, "domains": self.domains.saveAsDictionary()}

    @staticmethod
    def loadFromDictionary(d):
        domains = GeometrySelection.loadFromDictionary(d["domains"])
        return Source(d["name"], d["values"], domains=domains)

    @property
    def values(self):
        return self._value

    @values.setter
    def values(self, value):
        self._value = value
