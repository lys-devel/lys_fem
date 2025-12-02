import weakref
import gmsh
import numpy as np
import sympy as sp
from .base import FEMObject

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)


class GmshGeometry:
    _index = 0

    def __init__(self, fem, orders):
        self._fem = fem
        GmshGeometry._index += 1
        self._name = "Geometry" + str(GmshGeometry._index)
        self._model = self.__update(self._name, orders)
        self._orders = list(orders)

    def update(self, orders):
        self._model.remove(self._name)
        self.__update(self._name, orders)

    def __update(self, name, orders):
        model = gmsh.model()
        model.add(name)
        model.setCurrent(name)

        scale = TransGeom(self._fem)
        for order in orders:
            order.execute(model, scale)
        model.occ.removeAllDuplicates()
        model.occ.synchronize()
        for i in [1,2,3]:
            if len(model.getEntities(i))!=0:
                dim = i
        for i, obj in enumerate(model.getEntities(3)):
            model.add_physical_group(dim=3, tags=[obj[1]], tag=i + 1)
            model.setPhysicalName(dim=3, tag=i+1, name="domain"+str(i+1))
        for i, obj in enumerate(model.getEntities(2)):
            model.add_physical_group(dim=2, tags=[obj[1]], tag=i + 1)
            if dim == 2:
                model.setPhysicalName(dim=2, tag=i+1, name="domain"+str(i+1))
            elif dim == 3:
                model.setPhysicalName(dim=2, tag=i+1, name="boundary"+str(i+1))
        for i, obj in enumerate(model.getEntities(1)):
            model.add_physical_group(dim=1, tags=[obj[1]], tag=i + 1)
            if dim == 1:
                model.setPhysicalName(dim=1, tag=i+1, name="domain" + str(i+1))
            elif dim == 2:
                model.setPhysicalName(dim=1, tag=i+1, name="boundary" + str(i+1))
            else:
                model.setPhysicalName(dim=1, tag=i+1, name="edge" + str(i+1))
        for i, obj in enumerate(model.getEntities(0)):
            model.add_physical_group(dim=0, tags=[obj[1]], tag=i + 1)
            if dim == 1:
                model.setPhysicalName(dim=0, tag=i+1, name="boundary" + str(i+1))
            else:
                model.setPhysicalName(dim=0, tag=i+1, name="point" + str(i+1))
        return model

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        self._model.setCurrent(self._name)
        return self._model

    @property
    def mesh(self):
        self._model.setCurrent(self._name)
        return self._model.mesh
    
    @property
    def elementPositions(self):
        mesh = self.mesh
        dim = self._fem.dimension

        res = {}
        for etype, etags, enodes in zip(*mesh.getElements(dim=dim)):
            nnodes = mesh.getElementProperties(etype)[3]
            enodes = np.array(enodes).reshape(-1, nnodes)

            for elem_id, nodes in zip(etags, enodes):
                coords = [mesh.getNode(n)[0] for n in nodes]
                coords = np.array(coords).reshape(-1, 3).mean(axis=0)[:dim]
                res[elem_id] = coords
        return res

    def geometryParameters(self):
        self._model.setCurrent(self._name)
        result = {}
        for c in self._orders:
            result.update(c.generateParameters(self._model, TransGeom(self._fem)))
        return result

    def geometryAttributes(self, dim):
        self._model.setCurrent(self._name)
        return [tag for d, tag  in self._model.getPhysicalGroups(dim)]

    def duplicate(self):
        return GmshGeometry(self._fem, self._orders)

    def export(self, file):
        self._model.setCurrent(self._name)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(file)


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
            groups = {"Domain": {}, "Boundary": {}, "Volume": {}, "Surface": {}, "Edge": {}, "Point": {}}
        self._groups = groups
        self._default = None
        self._partial = None
        self._updated = True

    def add(self, command):
        self._order.append(command)
        command.setParent(self)
        self._updated=True

    def remove(self, command):
        self._order.remove(command)
        self._updated=True

    def addGroup(self, type, name, tags):
        self._groups[type][name] = tags

    def removeGroup(self, type, name):
        del self._groups[type][name]

    def _update(self):
        self._updated=True

    def generateGeometry(self, n=None):
        if n is None:
            if self._updated or self._default is None:
                self._default = GmshGeometry(self.fem, self._order)
            self._updated=False
            return self._default
        else:
            if self._partial is None:
                self._partial = GmshGeometry(self.fem, self._order[:n+1])
            else:
                self._partial.update(self._order[:n+1])
            return self._partial

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
        
    @scale.setter
    def scale(self, value):
        self._scale = value

    @property
    def commands(self):
        return self._order

    @property
    def groups(self):
        return self._groups

    def saveAsDictionary(self):
        return {"geometries": [c.saveAsDictionary() for c in self.commands], "scale": self._scale, "groups": self._groups}

    @staticmethod
    def loadFromDictionary(d):
        order = [FEMGeometry.loadFromDictionary(dic) for dic in d.get("geometries", [])]
        return GeometryGenerator(order, scale=d.get("scale", "auto"), groups=d.get("groups"))


class TransGeom:
    def __init__(self, fem):
        self._fem = fem

    def __call__(self, value, unit="1"):
        if unit == "m":
            scale = self._fem.geometries.scale
        else:
            scale = 1
        if isinstance(value, (list, tuple, np.ndarray)):
            return [self(v, unit) for v in value]
        elif isinstance(value, (float, int, sp.Float, sp.Integer)):
            return value/scale
        else:
            return value.subs(self._fem.parameters.getSolved())/scale


class FEMGeometry(object):
    def __init__(self, args):
        self._args = args
        self._parent = None

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value
        if self._parent is not None:
            self._parent()._update()

    def setParent(self, parent):
        self._parent = weakref.ref(parent)

    def saveAsDictionary(self):
        return {"type": self.type, "args": self.args}

    @staticmethod
    def loadFromDictionary(d):
        for t in sum(geometryCommands.values(), []):
            if t.type == d["type"]:
                return t(*d["args"])

    def generateParameters(self, model, scale):
        return {}


class GeometrySelection(FEMObject):
    def __init__(self, geometryType="Domain", selection=None, parent=None):
        if isinstance(selection, GeometrySelection):
            geometryType = selection.geometryType
            selection = selection.getSelection()
        if selection is None:
            selection = []
        self._geom = geometryType
        if isinstance(selection, str):
            self._selection = selection
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
        return [attr for attr in attrs if self._check(attr)]

    def _check(self, item):
        if self._selection == "all":
            return True
        elif isinstance(self._selection, str):
            return item in self.fem.geometries.groups[self._geom][self._selection]
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


geometryCommands = {}
