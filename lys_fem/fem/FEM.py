from .geometry import GeometryGenerator
from .mesh import OccMesher
from .material import MaterialList, Material


class FEMProject:
    def __init__(self, dim):
        self._dim = dim
        self._geom = GeometryGenerator(dim)
        self._mesher = OccMesher()
        self._materials = MaterialList([Material("Material1"), Material("Material2")])
        self._models = [ElasticModelGenerator()]

    def saveAsDictionary(self):
        d = {"dimension": self._dim}
        d["geometries"] = self._geom.saveAsDictionary()
        return d

    def loadFromDictionary(self, d):
        self._dim = d.get("dimension", 3)
        if "geometries" in d:
            self._geom.loadFromDictionary(d["geometries"])

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
        return self._modles

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

    def setVariableDimension(self, dim):
        self._nvar = dim
