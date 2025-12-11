import numpy as np

import gmsh
from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection
from lys_fem.geometry import GmshMesh


class OccMesher(FEMObject):
    _keys = {15: "point", 1: "line", 2: "triangle", 3: "quad", 4: "tetra", 5: "hexa", 6: "prism", 7: "pyramid"}

    def __init__(self, parent=None, refinement=0, transfinite=None, periodicity=None, size=None, file=None):
        super().__init__()
        self._current = "Default"
        if parent is not None:
            self.setParent(parent)
        self._refine = refinement
        if transfinite is None:
            transfinite=[]
        self._transfinite = FEMObjectList(self, transfinite)
        if size is None:
            size = []
        self._size = FEMObjectList(self, size)
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

    def getMeshWave(self, model, dim=3):
        from lys import Wave
        self.generate(model)
        model = model.model
        result = []
        for dim, grp in model.getPhysicalGroups(dim):
            coords_group = np.zeros((0, 3))
            nodes_group = np.zeros((0,), dtype=int)
            elements = {}
            for obj in model.getEntitiesForPhysicalGroup(dim, grp):
                coords, elem, nodes = self.__getMeshForEntity(model, dim, obj)
                coords_group = np.vstack([coords_group, np.reshape(coords, (-1, 3))])
                nodes_group = np.hstack([nodes_group, nodes], dtype=int)
                for type, nodetag in elem.items():
                    key = self._keys[type]
                    if key not in elements:
                        elements[key] = nodetag
                    else:
                        elements[key] = np.vstack([elements[key], nodetag])
            result.append(Wave(np.empty((coords_group.shape[0],)), coords_group, elements=elements, tag=grp, nodes=nodes_group))
        return result

    def __getMeshForEntity(self, model, dim, obj):
        nodes, coords, _ = model.mesh.getNodes(dim, obj, includeBoundary=True)
        sorter = np.argsort(nodes)
        types, _, nodetags = model.mesh.getElements(dim, obj)
        elem = {}
        for type, nodetag in zip(types, nodetags):
            nNodes = model.mesh.getElementProperties(type)[3]
            nodetag = sorter[np.searchsorted(nodes, nodetag, sorter=sorter)]
            nodetag = np.reshape(nodetag, (-1, nNodes))
            elem[type] = nodetag
        return np.reshape(coords, (-1, 3)), elem, nodes

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
