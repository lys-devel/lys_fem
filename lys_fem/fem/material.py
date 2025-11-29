import numpy as np

from lys_fem import util
from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection

materialParameters = {}


class Materials(FEMObjectList):
    def __init__(self, parent, materials):
        super().__init__(parent, materials)

    def append(self, material):
        if material.objName is None:
            names_used = [c.objName for c in self.parent.materials]
            i = 1
            while "Material" + str(i) in names_used:
                i += 1
            material.objName = "Material" + str(i)
        super().append(material)

    def eval(self, d):
        res = {}
        R = self._jacobi()
        for key, value in self._materialDict().items():
            res[key] = util.eval(value, d, name=key, geom="domain", J=R if key != "R" else None)
        return res

    def _materialDict(self):
        # find all parameter names and their groups
        groups = {}
        for m in self:
            for p in m:
                if p.name not in groups:
                    groups[p.name] = set()
                groups[p.name] |= set(p.getParameters().keys())

        # create coefficient for respective parameter
        res = {}
        for group, params in groups.items():
            for pname in params:
                res[pname] = self.__generateCoefForParameter(pname, group)
        return res

    def _jacobi(self):
        J = {"default": np.eye(3).tolist()}
        for m in self:
            R =  m.coordinate
            if R is not None:
                R = np.array([[float(rr) for rr in r] for r in R])
                for d in m.geometries:
                    J[d] = np.array([R[0]/np.linalg.norm(R[0]), R[1]/np.linalg.norm(R[1]), R[2]/np.linalg.norm(R[2])])
        if len(J) > 1:
            return util.eval(J, name="R", geom="domain")
        else:
            return None

    def __generateCoefForParameter(self, pname, group):
        coefs = {}
        for m in self:
            p = m[group]
            if p is not None:
                for d in m.geometries:
                    coefs[d] = p.getParameters()[pname]
        return coefs 


class Material(FEMObject):
    def __init__(self, params=None, geometries=None, objName=None, coord=None):
        super().__init__(objName)
        self._geometries = GeometrySelection("Domain", geometries, parent=self)
        if params is None:
            params = []
        self._params = params
        self._coord = coord

    def __getitem__(self, i):
        if isinstance(i, str):
            for p in self._params:
                if p.name == i:
                    return p
        else:
            return self._params.__getitem__(i)
        
    @property
    def parameters(self):
        return self._params

    @property
    def geometries(self):
        return self._geometries
    
    @property
    def coordinate(self):
        return self._coord
    
    @coordinate.setter
    def coordinate(self, value):
        self._coord = value

    @geometries.setter
    def geometries(self, value):
        self._geometries = GeometrySelection("Domain", value, parent=self)

    def saveAsDictionary(self):
        return {"name": self.objName, "geometries": self.geometries.saveAsDictionary(), "params": [p.saveAsDictionary() for p in self], "coord": self._coord}

    @staticmethod
    def loadFromDictionary(d):
        params = [FEMParameter.loadFromDictionary(p) for p in d["params"]]
        return Material(params, GeometrySelection.loadFromDictionary(d["geometries"]), objName=d["name"], coord=d.get("coord"))


class FEMParameter:
    def __init__(self, name):
        self._name = name

    def saveAsDictionary(self):
        d = dict(vars(self))
        for key, item in d.items():
            if isinstance(item, np.ndarray):
                d[key] = item.tolist()
        d["paramsName"] = self.name
        return d

    def getParameters(self):
        return {key: value for key, value in vars(self).items() if key[0] != "_" and value is not None}

    @staticmethod
    def loadFromDictionary(d):
        cls_list = set(sum(materialParameters.values(), []))
        cls_dict = {value.name: value for value in cls_list}

        d = dict(d)
        cls = cls_dict[d["paramsName"]]
        del d["paramsName"]
        return cls(**d)

    def widget(self, name):
        from lys.Qt import QtWidgets
        from ..widgets import ScalarFunctionWidget, MatrixFunctionWidget, VectorFunctionWidget
        param = self.getParameters()[name]
        if np.shape(param) == ():
            return ScalarFunctionWidget(None, getattr(self, name), valueChanged=lambda x: setattr(self, name, x))
        elif len(np.shape(param)) == 1:
            return VectorFunctionWidget(None, getattr(self, name), valueChanged=lambda x: setattr(self, name, x))
        elif len(np.shape(param)) == 2:
            return MatrixFunctionWidget(None, getattr(self, name), valueChanged=lambda x: setattr(self, name, x))
        else:
            return QtWidgets.QWidget()


class UserDefinedParameters(FEMParameter):
    name = "User Defined"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


materialParameters["User Defined"] = [UserDefinedParameters]