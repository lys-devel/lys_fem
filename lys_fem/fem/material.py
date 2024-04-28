
from .base import FEMObject, FEMObjectList, FEMCoefficient, exprToStr, strToExpr
from .geometry import GeometrySelection

materialParameters = {}

def  _getParameters(name=None):
    """
    Get dictionary that contains material parameter classes.
    Example: {"Heat Conduction": HeatConductionParameters, "Elasticity": ElasticityParameters}
    """
    cls_list = set(sum(materialParameters.values(), []))
    cls_dict = {value.name: value for value in cls_list}
    if name is None:
        return cls_dict
    else:
        return cls_dict[name]


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

    def defaultParameter(self, groupName, dim):
        default = _getParameters(groupName)()
        return default.getParameters(dim)

    def materialDict(self, dim):
        # find all parameter names and their groups
        groups = {}
        for m in self:
            for p in m:
                if p.name in groups:
                    groups[p.name] |= set(p.getParameters(dim).keys())
                else:
                    groups[p.name] = set(p.getParameters(dim).keys())

        # create coefficient for respective parameter
        res = {}
        for group, params in groups.items():
            for pname in params:
                res[pname] = self.__generateCoefForParameter(pname, group, dim)
        return res

    def __generateCoefForParameter(self, pname, group, dim):
        #coefs = {"default": self.defaultParameter(group, dim)[pname]}
        scale = self.fem.scaling.getScaling(_getParameters(group).units[pname])
        coefs = FEMCoefficient(geomType="Domain", scale=scale, xscale=self.fem.scaling.getScaling("m"), vars=self.fem.parameters.getSolved())
        for m in self:
            p = m[group]
            if p is not None:
                for d in m.geometries:
                    coefs[d] = p.getParameters(dim)[pname]
        return coefs 


class Material(FEMObject):
    def __init__(self, params=None, geometries=None, objName=None):
        super().__init__(objName)
        self._geometries = GeometrySelection("Domain", geometries, parent=self)
        if params is None:
            params = []
        self._params = params

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

    @geometries.setter
    def geometries(self, value):
        self._geometries = GeometrySelection("Domain", value, parent=self)

    def saveAsDictionary(self):
        return {"name": self.objName, "geometries": self.geometries.saveAsDictionary(), "params": [p.saveAsDictionary() for p in self]}

    @staticmethod
    def loadFromDictionary(d):
        params = [FEMParameter.loadFromDictionary(p) for p in d["params"]]
        return Material(params, GeometrySelection.loadFromDictionary(d["geometries"]), objName=d["name"])


class FEMParameter:
    def __init__(self, name):
        self._name = name

    def saveAsDictionary(self):
        d = dict(vars(self))
        for key, value in d.items():
            d[key] = exprToStr(value)
        d["paramsName"] = self.name
        return d

    def getParameters(self):
        return vars(self)

    @staticmethod
    def loadFromDictionary(d):
        cls_list = set(sum(materialParameters.values(), []))
        cls_dict = {value.name: value for value in cls_list}

        d = dict(d)
        cls = cls_dict[d["paramsName"]]
        del d["paramsName"]
        print(d)
        for key, value in d.items():
            d[key] = strToExpr(value)
        print(d)
        return cls(**d)
