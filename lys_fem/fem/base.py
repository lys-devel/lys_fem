import weakref
import numpy as np
import sympy as sp

class FEMObject:
    def __init__(self, objName=None):
        self._objName = objName

    @property
    def objName(self):
        return self._objName
    
    @objName.setter
    def objName(self, value):
        self._objName = value

    @classmethod
    @property
    def className(cls):
        raise NotImplementedError

    @property
    def fem(self):
        from .FEM import FEMProject
        obj = self
        while not isinstance(obj, FEMProject):
            obj = obj.parent
        return obj

    @property
    def parent(self):
        return self._parent()

    def setParent(self, parent):
        self._parent = weakref.ref(parent)


class FEMObjectList(list, FEMObject):
    def __init__(self, parent, items=[]):
        super().__init__(items)
        self.setParent(parent)
        for item in items:
            if isinstance(item, FEMObject):
                item.setParent(self)

    def append(self, item):
        super().append(item)
        item.setParent(self)


class FEMCoefficient(dict):
    def __init__(self, value={}, geomType="Domain", scale=1, xscale=1, vars={}):
        super().__init__()
        self._type = geomType
        self._scale = scale
        self._xscale = xscale
        self._vars = vars
        for key, val in value.items():
            self[key] = val

    @property
    def geometryType(self):
        return self._type
    
    @property
    def scale(self):
        return self._scale
    
    @property
    def xscale(self):
        return self._xscale

    def __setitem__(self, key, value):
        super().__setitem__(key, self.__parseItem(value))

    def __parseItem(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [self.__parseItem(v) for v in value]
        elif isinstance(value, (int, float, sp.Integer, sp.Float)):
            return value/self._scale
        elif isinstance(value, sp.Basic):
            xs, ys, zs = sp.symbols("x_scaled,y_scaled,z_scaled")
            val = value.subs({"x":xs*self._xscale, "y":ys*self._xscale, "z":zs*self._xscale})/self._scale
            return val.subs(self._vars)
        else:
            return value


def strToExpr(x):
    from .conditions import CalculatedResult
    if x.startswith("[String]"):
        return x.replace("[String]","")
    if x.startswith("[CalculatedResult]"):
        x = x.replace("[CalculatedResult]", "")
        return CalculatedResult.loadFromDictionary(eval(x))
    try:
        res = eval(x,{})
    except:
        res = sp.parsing.sympy_parser.parse_expr(x)
    return res


def exprToStr(x):
    from .conditions import CalculatedResult
    if isinstance(x, str):
        return "[String]"+x
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, CalculatedResult):
        x = "[CalculatedResult]" + str(x.saveAsDictionary())
    return str(x)