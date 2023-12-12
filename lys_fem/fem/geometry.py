import gmsh
from .base import FEMObject

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)


class GeometryGenerator:
    def __init__(self, order=None):
        super().__init__()
        if order is None:
            order = []
        self._order = order

    def add(self, command):
        self._order.append(command)

    def generateGeometry(self, n=None):
        model = gmsh.model()
        model.add("Default")
        model.setCurrent("Default")
        for order in self._order if n is None else self._order[0:n + 1]:
            order.execute(model)
        model.occ.removeAllDuplicates()
        model.occ.synchronize()
        for i, obj in enumerate(model.getEntities(3)):
            model.add_physical_group(dim=3, tags=[obj[1]], tag=i + 1)
            model.setPhysicalName(dim=3, tag=i+1, name=str(i+1))
        for i, obj in enumerate(model.getEntities(2)):
            model.add_physical_group(dim=2, tags=[obj[1]], tag=i + 1)
            model.setPhysicalName(dim=2, tag=i+1, name=str(i+1))
        for i, obj in enumerate(model.getEntities(1)):
            model.add_physical_group(dim=1, tags=[obj[1]], tag=i + 1)
            model.setPhysicalName(dim=1, tag=i+1, name=str(i+1))
        for i, obj in enumerate(model.getEntities(0)):
            model.add_physical_group(dim=0, tags=[obj[1]], tag=i + 1)
            model.setPhysicalName(dim=0, tag=i+1, name=str(i+1))
        return model

    def geometryAttributes(self, dim):
        m = self.generateGeometry()
        return [tag for d, tag  in m.getPhysicalGroups(dim)]

    @property
    def commands(self):
        return self._order

    def saveAsDictionary(self):
        return {"geometries": [c.saveAsDictionary() for c in self.commands]}

    @staticmethod
    def loadFromDictionary(d):
        order = [FEMGeometry.loadFromDictionary(dic) for dic in d.get("geometries", [])]
        return GeometryGenerator(order)


class FEMGeometry(object):
    def saveAsDictionary(self):
        return {"type": self.type, "args": self.args}

    @staticmethod
    def loadFromDictionary(d):
        for t in sum(geometryCommands.values(), []):
            if t.type == d["type"]:
                return t(*d["args"])


class GeometrySelection(FEMObject):
    def __init__(self, geometryType="Domain", selection=None):
        if selection is None:
            selection = []
        self._geom = geometryType
        if isinstance(selection, str):
            self._selection = selection
        else:
            self._selection = list(selection)

    def check(self, item):
        if self._selection == "all":
            return True
        else:
            return item in self._selection

    def getSelection(self, geom=None):
        if geom is None:
            return self._selection

    def setSelection(self, value):
        self._selection = value

    def append(self, value):
        self._selection.append(value)
        self._selection = sorted(self._selection)

    def remove(self, value):
        self._selection.remove(value)

    def selectionType(self):
        if self._selection == "all":
            return "All"
        else:
            return "Selected"

    def __iter__(self):
        return self.geometryAttributes.__iter__()
        
    @property
    def geometryAttributes(self):
        if self._geom == "Domain":
            return [attr for attr in self.fem.domainAttributes if self.check(attr)]
        else:
            return [attr for attr in self.fem.boundaryAttributes if self.check(attr)]

    @property
    def geometryType(self):
        return self._geom

    def saveAsDictionary(self):
        return {"selection": self._selection, "geometryType": self._geom}

    @staticmethod
    def loadFromDictionary(d):
        return GeometrySelection(d["geometryType"], d["selection"])


geometryCommands = {}
