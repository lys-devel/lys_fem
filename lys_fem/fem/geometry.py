import gmsh
from lys.Qt import QtCore

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)


class GeometryGenerator(QtCore.QObject):
    updated = QtCore.pyqtSignal()

    def __init__(self, order=None):
        super().__init__()
        if order is None:
            order = []
        self._order = order

    def addCommand(self, command):
        self._order.append(command)
        self.updated.emit()

    def generateGeometry(self, n=None):
        model = gmsh.model()
        model.add("Default")
        model.setCurrent("Default")
        for order in self._order if n is None else self._order[0:n + 1]:
            order.execute(model)
        model.occ.removeAllDuplicates()
        model.occ.synchronize()
        for obj in model.getEntities(3):
            model.add_physical_group(dim=3, tags=[obj[1]])
        for obj in model.getEntities(2):
            model.add_physical_group(dim=2, tags=[obj[1]])
        for obj in model.getEntities(1):
            model.add_physical_group(dim=1, tags=[obj[1]])
        return model

    @property
    def commands(self):
        return self._order

    def saveAsDictionary(self):
        return {"geometries": [c.saveAsDictionary() for c in self.commands]}

    @staticmethod
    def loadFromDictionary(d):
        order = [GeometryOrder.loadFromDictionary(dic) for dic in d.get("geometries", [])]
        return GeometryGenerator(order)


class FEMGeometry(object):
    def saveAsDictionary(self):
        return {"type": self.type, "args": self.args}

    @staticmethod
    def loadFromDictionary(d):
        for t in sum(geometryCommands.values(), []):
            if t.type == d["type"]:
                return t(*d["args"])


geometryCommands = {}
