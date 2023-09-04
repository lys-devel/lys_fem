import gmsh
import numpy as np
from lys import Wave
from lys.Qt import QtCore

from .geometryOrder import AddRect


def _initialize():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)


class GeometryGenerator(QtCore.QObject):
    updated = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._order = []
        self.addCommand(AddRect.default)

    def addCommand(self, command):
        self._order.append(command)
        self.updated.emit()

    def generateGeometry(self, n=None):
        model = gmsh.model()
        model.add("Default")
        model.setCurrent("Default")
        for order in self._order[0:] if n is None else self._order[0:n + 1]:
            order.execute(model)
        model.occ.synchronize()
        for obj in model.getEntities(3):
            model.add_physical_group(dim=3, tags=[obj[1]])
        for obj in model.getEntities(2):
            model.add_physical_group(dim=2, tags=[obj[1]])
        return OccGeometry(model)

    @property
    def commands(self):
        return self._order


class OccGeometry(QtCore.QObject):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def getMeshWave(self, dim=3):
        self._model.mesh.generate(dim)

        result = []
        for dim, grp in self._model.getPhysicalGroups(dim):
            coords_group = np.zeros((0, 3))
            elements = {}
            for obj in self._model.getEntitiesForPhysicalGroup(dim, grp):
                coords, elem = self.__getMeshForEntity(dim, obj)
                coords_group = np.vstack([coords_group, np.reshape(coords, (-1, 3))])
                for type, nodetag in elem.items():
                    if type not in elements:
                        elements[type] = nodetag
                    else:
                        elements[type] = np.vstack([elements[type], nodetag])
            result.append(Wave(np.empty((coords_group.shape[0],)), coords_group, elements=elements, tag=grp))
        return result

    def __getMeshForEntity(self, dim, obj):
        nodes, coords, _ = self._model.mesh.getNodes(dim, obj, includeBoundary=True)
        sorter = np.argsort(nodes)
        types, _, nodetags = self._model.mesh.getElements(dim, obj)
        elem = {}
        for type, nodetag in zip(types, nodetags):
            nNodes = self._model.mesh.getElementProperties(type)[3]
            nodetag = sorter[np.searchsorted(nodes, nodetag, sorter=sorter)]
            nodetag = np.reshape(nodetag, (-1, nNodes))
            elem[type] = nodetag
        return np.reshape(coords, (-1, 3)), elem


_initialize()
