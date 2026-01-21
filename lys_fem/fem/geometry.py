import json
from lys_fem.geometry import FEMGeometry, GmshGeometry, GeometrySelection
from .base import FEMObject


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
        self._groups[name] = GeometrySelection(type, value)

    def removeGroup(self, name):
        del self._groups[name]

    def _update(self):
        self._updated=True

    def generateGeometry(self, n=None):
        if n is None:
            if self._updated or self._default is None:
                self._default = GmshGeometry(self._order, groups=self._groups, params=self.fem.parameters.getSolved())
            self._updated=False
            return self._default
        else:
            return GmshGeometry(self._order[:n+1], groups=self._groups, params=self.fem.parameters.getSolved())

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

