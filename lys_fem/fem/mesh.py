import itertools
import numpy as np

import gmsh
from lys import Wave


class OccMesher(object):
    _keys = {1: "line", 2: "triangle", 3: "quad", 4: "tetra", 5: "hexa", 6: "prism", 7: "pyramid"}

    def __init__(self):
        super().__init__()
        self._refine = 0
        self._partialRefine = {}

    def setPartialRefinement(self, dim, tag, factor):
        self._partialRefine[(dim, tag)] = factor

    def setRefinement(self, n):
        self._refine = n

    def _generate(self, model):
        # prepare partial refinement
        self._refineData = []
        if len(self._partialRefine) != 0:
            for (dim, tag), factor in self._partialRefine.items():
                if dim == 1:
                    tagSet = set([tag])
                if dim == 2:
                    _, edges = model.getAdjacencies(2, tag)
                    tagSet = set(edges)
                if dim == 3:
                    _, surfs = model.getAdjacencies(3, tag)
                    edges = itertools.chain.from_iterable([model.getAdjacencies(2, v)[1] for v in surfs])
                    tagSet = set(edges)
                self._refineData.append((tagSet, factor))
            model.mesh.setSizeCallback(self.__callback)
        # generate and refine
        model.mesh.setTransfiniteAutomatic()
        model.mesh.generate()
        for _ in range(self._refine):
            model.mesh.refine()

    def __callback(self, dim, tag, x, y, z, lc):
        if dim == 1:
            for edges, factor in self._refineData:
                if tag in edges:
                    lc = lc / factor
        return lc

    def getMeshWave(self, model, dim=3):
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

    def export(self, model, file):
        self._generate(model)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(file)
