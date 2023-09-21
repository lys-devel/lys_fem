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


class GeometryOrder(object):
    def saveAsDictionary(self):
        return {"type": self.type, "args": self.args}

    @staticmethod
    def loadFromDictionary(d):
        for t in sum(geometryCommands.values(), []):
            if t.type == d["type"]:
                return t(*d["args"])


class AddBox(GeometryOrder):
    def __init__(self, x=0, y=0, z=0, dx=1, dy=1, dz=1):
        self.args = [x, y, z, dx, dy, dz]

    def execute(self, model):
        model.occ.addBox(*self.args)

    @classmethod
    @property
    def type(cls):
        return "Add box"


class AddRect(GeometryOrder):
    def __init__(self, x=0, y=0, z=0, dx=1, dy=1):
        self.args = [x, y, z, dx, dy]

    def execute(self, model):
        model.occ.addRectangle(*self.args)

    @classmethod
    @property
    def type(cls):
        return "Add rectangle"


class AddLine(GeometryOrder):
    def __init__(self, x1=0, y1=0, z1=0, x2=1, y2=0, z2=0):
        self.args = (x1, y1, z1, x2, y2, z2)

    def execute(self, model):
        p1t = model.occ.addPoint(*self.args[:3])
        p2t = model.occ.addPoint(*self.args[3:])
        model.occ.addLine(p1t, p2t)

    @classmethod
    @property
    def type(cls):
        return "Add line"


geometryCommands = {"Add 3D": [AddBox], "Add 2D": [AddRect], "Add 1D": [AddLine]}
