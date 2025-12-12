
from lys_fem.geometry import GmshMesh
from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection


class OccMesher(FEMObject):
    def __init__(self, parent=None, refinement=0, transfinite=None, periodicity=None, size=None, file=None):
        super().__init__()
        self._current = "Default"
        if parent is not None:
            self.setParent(parent)
        self._refine = refinement
        if transfinite is None:
            transfinite=[]
        self._transfinite = FEMObjectList(transfinite, parent=self)
        if size is None:
            size = []
        self._size = FEMObjectList(size, parent=self)
        if periodicity is None:
            periodicity = []
        self._periodicity = periodicity
        self._file = file
        self._duplicated_model = None

    def addTransfinite(self, geomType="Volume", geometries=[]):
        geom = GeometrySelection(geometryType=geomType, selection=geometries)
        self._transfinite.append(geom)
        return geom

    @property
    def transfinite(self):
        return self._transfinite

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, value):
        self._file = value

    @property
    def refinement(self):
        """
        Get refinement factor.

        Returns:
            int: The refinement factor.
        """
        return self._refine

    def setRefinement(self, n):
        """
        Set refinement factor.

        Args:
            n(int): The refinement factor.
        """
        self._refine = n

    @property
    def sizeConstraint(self):
        return self._size

    def addSizeConstraint(self, geomType="Volume", geometries=[], size=1):
        geom = GeometrySelection(geometryType=geomType, selection=geometries)
        geom.size = size
        self._size.append(geom)
        return geom

    @property
    def periodicPairs(self):
        return self._periodicity

    def generate(self, model):
        return GmshMesh(model, self._transfinite, self._periodicity, self._size, self._refine)

    def saveAsDictionary(self):
        pairs = [(p[0].saveAsDictionary(), p[1].saveAsDictionary()) for p in self._periodicity]
        size = [{"size": p.size, "geometries": p.saveAsDictionary()} for p in self._size]
        trans = [t.saveAsDictionary() for t in self._transfinite]
        return {"refine": self.refinement, "periodicity": pairs, "transfinite": trans, "size": size, "file": self._file}

    @classmethod
    def loadFromDictionary(cls, d):
        pairs = [(GeometrySelection.loadFromDictionary(p[0]), GeometrySelection.loadFromDictionary(p[1])) for p in d.get("periodicity", [])]
        size = []
        for p in d.get("size", []):
            g = GeometrySelection.loadFromDictionary(p["geometries"])
            g.size = p["size"]
            size.append(g)
        transfinite=[]
        for p in d.get("transfinite", []):
            g = GeometrySelection.loadFromDictionary(p)
            transfinite.append(g)
        return OccMesher(refinement=d["refine"], transfinite=transfinite, periodicity=pairs, size=size, file=d.get("file", None))
