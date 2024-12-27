import itertools
import numpy as np

import gmsh
from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection


class OccMesher(FEMObject):
    _keys = {15: "point", 1: "line", 2: "triangle", 3: "quad", 4: "tetra", 5: "hexa", 6: "prism", 7: "pyramid"}

    def __init__(self, parent=None, refinement=0, partialRefine=None, transfinite=None, periodicity=None, size=None, file=None):
        super().__init__()
        self._current = "Default"
        if parent is not None:
            self.setParent(parent)
        self._refine = refinement
        if partialRefine is None:
            partialRefine = []
        self._partialRefine = FEMObjectList(self, partialRefine)
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

    def addTransfinite(self, geomType="Volume", geometries=[]):
        geom = GeometrySelection(geometryType=geomType, selection=geometries)
        self._transfinite.append(geom)
        return geom

    @property
    def transfinite(self):
        return self._transfinite

    def setPartialRefinement(self, dim, tag, factor):
        self._partialRefine[(dim, tag)] = factor

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
    def partialRefinement(self):
        return self._partialRefine

    def addPartialRefinement(self, geomType="Volume", geometries=[], factor=1):
        geom = GeometrySelection(geometryType=geomType, selection=geometries)
        geom.factor = factor
        self._partialRefine.append(geom)
        return geom

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

    def _generate(self, model):
        model.mesh.clear()
        
        if self._file is not None:
            gmsh.merge(self._file)
            return
   
        self.__setTransfinite(model)
        self.__setPeriodicity(model)
        self.__partialRefine(model)
        self.__setSize(model)

        model.mesh.generate()
        for _ in range(self._refine):
            model.mesh.refine()

        model.mesh.optimize()

    def __setTransfinite(self, model):
        # prepare partial refinement
        for geom in self._transfinite:
            dim = {"Volume": 3, "Surface": 2, "Edge": 1, "Point": 0}[geom.geometryType]
            for tag in geom:
                if dim == 2:
                    surf = model.getEntitiesForPhysicalGroup(2, tag)[0]
                    model.mesh.setTransfiniteSurface(surf)
                    model.mesh.setRecombine(2, surf)
                if dim == 3:
                    domain = model.getEntitiesForPhysicalGroup(3, tag)[0]
                    _, surfs = model.getAdjacencies(3, domain)
                    for v in surfs:
                        model.mesh.setTransfiniteSurface(v)
                        model.mesh.setRecombine(2, v)
                    model.mesh.setTransfiniteVolume(domain)

    def __setPeriodicity(self, model):
        if len(self._periodicity) == 0:
            return
        if len(model.getPhysicalGroups(3)) != 0:
            sdim = 2
        else:
            sdim = 1
        ent = [tag for dim, tag in model.getEntities(sdim)]
        for pair in self._periodicity:
            for p1, p2 in zip(pair[0].getSelection(), pair[1].getSelection()):
                t = self.__getTransform(model, sdim, ent[p1 - 1], ent[p2 - 1])
                model.mesh.setPeriodic(sdim, [ent[p1 - 1]], [ent[p2 - 1]], t)

    def __getTransform(self, model, sdim, e1, e2):
        """
        Consider only parallel shift.
        """
        c1 = np.array([model.getValue(*obj, []) for obj in model.getBoundary([(sdim, e1)], recursive=True)])
        c2 = np.array([model.getValue(*obj, []) for obj in model.getBoundary([(sdim, e2)], recursive=True)])
        for c in c2:
            shift = c1[0] - c
            dist = max([min(np.linalg.norm(c2 - (d - shift), axis=1)) for d in c1])
            if dist < 1e-3:
                return [1, 0, 0, shift[0], 0, 1, 0, shift[1], 0, 0, 1, shift[2], 0, 0, 0, 1]
            
    def __partialRefine(self, model):
        # prepare partial refinement
        self._refineData = []
        if len(self._partialRefine) != 0:
            for geom in self._partialRefine:
                dim = {"Volume": 3, "Surface": 2, "Edge": 1, "Point": 0}[geom.geometryType]
                tagSet = set()
                for tag in geom:
                    if dim == 1:
                        edge = model.getEntitiesForPhysicalGroup(1, tag)[0]
                        tagSet.add(edge)
                    if dim == 2:
                        surf = model.getEntitiesForPhysicalGroup(2, tag)[0]
                        _, edges = model.getAdjacencies(2, surf)
                        tagSet |= set(edges)
                    if dim == 3:
                        domain = model.getEntitiesForPhysicalGroup(3, tag)[0]
                        _, surfs = model.getAdjacencies(3, domain)
                        edges = itertools.chain.from_iterable([model.getAdjacencies(2, v)[1] for v in surfs])
                        tagSet |= set(edges)
                self._refineData.append((tagSet, geom.factor))
            model.mesh.setSizeCallback(self.__callback)

    def __callback(self, dim, tag, x, y, z, lc):
        if dim == 1:
            for edges, factor in self._refineData:
                if tag in edges:
                    lc = lc / (factor+1)
        return lc

    def __setSize(self, model):
        # size constraint
        for geom in sorted(self._size, key=lambda x: 1/x.size):
            dim = {"Volume": 3, "Surface": 2, "Edge": 1, "Point": 0}[geom.geometryType]
            for tag in geom:
                if dim == 0:
                    model.mesh.setSize((dim, tag), geom.size)
                else:
                    ents = [(dim, t) for t in model.getEntitiesForPhysicalGroup(dim, tag)]
                    model.mesh.setSize(model.getBoundary(ents, recursive=True), geom.size/self.fem.geometries.scale)

    def getMeshWave(self, model, dim=3):
        from lys import Wave
        self._generate(model)
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

    def export(self, model, file, nogen=False):
        if not nogen:
            self._generate(model)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(file)

    def exportRefinedMesh(self, model, elems, size, file, amr):
        if self._current == "Default":
            present = "Default"
            next = "Refined"
        else:
            present = "Refined"
            next = "Default"

        model.setCurrent(next)

        alpha, nodes, size = self.__refine(model, elems, size, amr, present, 1)

        n = 0
        while nodes < amr.nodes*0.95 or nodes > amr.nodes*1.05:
            alpha, nodes, size = self.__refine(model, elems, size, amr, present, alpha)
            n += 1
            if n > 30:
                break
        self._current = next

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(file)

    def __refine(self, model, elems, size, amr, present, alpha):
        size = size*alpha
        size[size < amr.range[0]] = amr.range[0]
        size[size > amr.range[1]] = amr.range[1]
        view = gmsh.view.add("refinement")
        gmsh.view.addModelData(view, 0, present, "ElementData", elems, size)

        model.mesh.clear()
        field = model.mesh.field.add("PostView")
        model.mesh.field.setNumber(field, "ViewTag", view)
        model.mesh.field.setAsBackgroundMesh(field)
        model.mesh.generate()
        nodes = len(model.mesh.getNodes()[0])
        return (nodes/amr.nodes)**(1/self.fem.dimension), nodes, size

    def saveAsDictionary(self):
        pairs = [(p[0].saveAsDictionary(), p[1].saveAsDictionary()) for p in self._periodicity]
        partial = [{"factor": p.factor, "geometries": p.saveAsDictionary()} for p in self._partialRefine]
        size = [{"size": p.size, "geometries": p.saveAsDictionary()} for p in self._size]
        trans = [t.saveAsDictionary() for t in self._transfinite]
        return {"refine": self.refinement, "partial": partial, "periodicity": pairs, "transfinite": trans, "size": size, "file": self._file}

    @classmethod
    def loadFromDictionary(cls, d):
        pairs = [(GeometrySelection.loadFromDictionary(p[0]), GeometrySelection.loadFromDictionary(p[1])) for p in d.get("periodicity", [])]
        partial = []
        for p in d["partial"]:
            g = GeometrySelection.loadFromDictionary(p["geometries"])
            g.factor = p["factor"]
            partial.append(g)
        size = []
        for p in d.get("size", []):
            g = GeometrySelection.loadFromDictionary(p["geometries"])
            g.size = p["size"]
            size.append(g)
        transfinite=[]
        for p in d.get("transfinite", []):
            g = GeometrySelection.loadFromDictionary(p)
            transfinite.append(g)
        return OccMesher(refinement=d["refine"], partialRefine=partial, transfinite=transfinite, periodicity=pairs, size=size, file=d.get("file", None))
