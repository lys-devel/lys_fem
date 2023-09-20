from .geometry import GeometryGenerator
from .mesh import OccMesher
from .material import MaterialList, Material


class FEMProject:
    def __init__(self, dim):
        self._dim = dim
        self._geom = GeometryGenerator(dim)
        self._mesher = OccMesher()
        self._materials = MaterialList([Material("Material1", domains="all")])
        self._models = [ElasticModelGenerator()]

    def saveAsDictionary(self):
        d = {"dimension": self._dim}
        d["geometries"] = self._geom.saveAsDictionary()
        d["materials"] = self._materials.saveAsDictionary()
        return d

    def loadFromDictionary(self, d):
        self._dim = d.get("dimension", 3)
        if "geometries" in d:
            self._geom.loadFromDictionary(d["geometries"])
        if "materials" in d:
            self._materials.loadFromDictionary(d["materials"])

    @property
    def dimension(self):
        return self._dim

    @property
    def geometryGenerator(self):
        return self._geom

    @property
    def mesher(self):
        return self._mesher

    @property
    def models(self):
        return self._models

    def getMeshWave(self, dim=None, nomesh=False):
        if dim is None:
            dim = self._dim
        if nomesh:
            mesher = OccMesher()
        else:
            mesher = self._mesher
        return mesher.getMeshWave(self._geom.generateGeometry(), dim=dim)


class ElasticModelGenerator:
    def __init__(self):
        self._nvar = 3
        self._init = [InitialCondition("Default", self._nvar, domains="all")]

    def setVariableDimension(self, dim):
        self._nvar = dim

    def variableDimension(self):
        return self._nvar

    @property
    def initialConditions(self):
        return self._init

    def addInitialCondition(self, init=None):
        if init is None:
            i = 0
            while "Initial value" + str(i) not in [c.name for c in self._init]:
                i += 1
            name = "Initial value" + str(i)
            init = InitialCondition(name, self._nvar)


class InitialCondition:
    def __init__(self, name, nvar, value=None, domains=None):
        self._name = name
        if value is None:
            value = ["0"] * nvar
        self._value = value
        if domains is None:
            domains = []
        self._domains = domains

    def setDimension(self, dim):
        if len(self._value) > dim:
            self._value = self._value[:2]
        while len(self._value) < dim:
            self._value.append("0")

    @property
    def name(self):
        return self._name

    @property
    def domains(self):
        return self._domains

    @property
    def values(self):
        return self._value
