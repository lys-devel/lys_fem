import json
from .base import FEMObject
from lys_fem.geometry import FEMGeometry, GmshGeometry


class GeometryGenerator(FEMObject):
    def __init__(self, order=None, scale="auto", groups=None):
        super().__init__()
        self._scale = scale
        if order is None:
            order = []
        self._order = []
        for ord in order:
            self.add(ord)
        if groups is None:
            groups = {}
        self._groups = groups
        for value in groups.values():
            value.setParent(self)
        self._default = None
        self._updated = True

    def add(self, command):
        if hasattr(command, "__iter__"):
            for c in command:
                self.add(c)
            return
        self._order.append(command)
        command.setCallback(self._update)
        self._updated=True

    def remove(self, command):
        self._order.remove(command)
        self._updated=True

    def clear(self):
        for ord in self._order:
            self.remove(ord)

    def addGroup(self, type, name, value=[]):
        self._groups[name] = GeometrySelection(type, value, parent=self)

    def removeGroup(self, name):
        del self._groups[name]

    def _update(self):
        self._updated=True

    def generateGeometry(self, n=None):
        if n is None:
            if self._updated or self._default is None:
                self._default = GmshGeometry(self._order, params=self.fem.parameters.getSolved())
            self._updated=False
            return self._default
        else:
            return GmshGeometry(self._order[:n+1], params=self.fem.parameters.getSolved())

    def geometryParameters(self):
        return self.generateGeometry().geometryParameters()

    def geometryAttributes(self, dim):
        return self.generateGeometry().geometryAttributes(dim)

    @property
    def scale(self):
        if self._scale == "auto":
            def flatten(x):
                for item in x:
                    if hasattr(item, "__iter__"):
                        yield from flatten(item)
                    else:
                        yield abs(item)
            args = flatten([cc for c in self.commands for cc in c.args])
            return min([arg for arg in args if arg!=0])
        else:
            return self._scale
        
    @property
    def commands(self):
        return self._order

    @property
    def groups(self):
        return self._groups

    def saveAsDictionary(self):
        groups = {key: g.saveAsDictionary() for key, g in self._groups.items()}
        return {"geometries": [c.saveAsDictionary() for c in self.commands], "scale": self._scale, "groups": groups}

    @staticmethod
    def loadFromDictionary(d):
        order = [FEMGeometry.loadFromDictionary(dic) for dic in d.get("geometries", [])]
        groups = {key: GeometrySelection.loadFromDictionary(val) for key, val in d.get("groups", {}).items()}
        return GeometryGenerator(order, scale=d.get("scale", "auto"), groups=groups)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.saveAsDictionary(), f)

    def load(self, path):
        self.clear()
        with open(path) as f:
            g = GeometryGenerator.loadFromDictionary(json.load(f))
        self._scale = g._scale
        self._groups = g._groups
        for order in g.commands:
            self.add(order)



class GeometrySelection(FEMObject):
    def __init__(self, geometryType="Domain", selection=None, parent=None):
        if isinstance(selection, GeometrySelection):
            geometryType = selection.geometryType
            selection = selection.getSelection()
        if selection is None:
            selection = []
        self._geom = geometryType
        if selection == "all":
            self._selection = "all"
        elif isinstance(selection, str):
            self._selection = [selection]
        else:
            self._selection = list(selection)
        if parent is not None:
            self.setParent(parent)
        
    def __getitem__(self, index):
        return self._selection[index]

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

    def clear(self):
        self._selection.clear()

    def selectionType(self):
        if self._selection == "all":
            return "All"
        else:
            if len(self._selection) == 0:
                return "Group"
            elif isinstance(self._selection[0], str):
                return "Group"
            else:
                return "Selected"

    def __iter__(self):
        return self.geometryAttributes.__iter__()
        
    @property
    def geometryAttributes(self):
        if self._geom == "Domain":
            attrs = self.fem.domainAttributes
        elif self._geom == "Boundary":
            attrs = self.fem.boundaryAttributes
        elif self._geom == "Volume":
            attrs = self.fem.geometries.geometryAttributes(3)
        elif self._geom == "Surface":
            attrs = self.fem.geometries.geometryAttributes(2)
        elif self._geom == "Edge":
            attrs = self.fem.geometries.geometryAttributes(1)
        elif self._geom == "Point":
            attrs = self.fem.geometries.geometryAttributes(0)
        return [attr for attr in attrs if self.check(attr)]

    def check(self, item):
        if self._selection == "all":
            return True
        if len(self._selection) == 0:
            return False
        if isinstance(self._selection[0], str):
            return any([item in self.fem.geometries.groups[s] for s in self._selection])
        else:
            return item in self._selection

    @property
    def geometryType(self):
        return self._geom

    def saveAsDictionary(self):
        return {"selection": self._selection, "geometryType": self._geom}

    @staticmethod
    def loadFromDictionary(d):
        return GeometrySelection(d["geometryType"], d["selection"])

