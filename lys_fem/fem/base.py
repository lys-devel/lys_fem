import weakref
import numpy as np

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
    def model(self):
        from .model import FEMModel
        obj = self
        while not isinstance(obj, FEMModel):
            obj = obj.parent
        return obj

    @property
    def parent(self):
        return self._parent()

    def setParent(self, parent):
        self._parent = weakref.ref(parent)


class FEMObjectList(list, FEMObject):
    def __init__(self, items=[], parent=None, objName=None):
        super().__init__(items)
        FEMObject.__init__(self, objName)
        if parent is not None:
            self.setParent(parent)
        for item in items:
            if isinstance(item, FEMObject):
                item.setParent(self)

    def append(self, item):
        super().append(item)
        item.setParent(self)


class FEMObjectDict(dict, FEMObject):
    def __init__(self, items={}, parent=None, objName=None):
        super().__init__(items)
        FEMObject.__init__(self, objName)
        if parent is not None:
            self.setParent(parent)
        for item in items.values():
            if isinstance(item, FEMObject):
                item.setParent(self)

    def __setitem__(self, key, item):
        super().__setitem__(key, item)
        if isinstance(item, FEMObject):
            item.setParent(self)


class Coef(FEMObject):
    def __init__(self, expr, shape=(), description="", default=None):
        super().__init__()
        if isinstance(expr, np.ndarray):
            expr = expr.tolist()
        self._expression = expr
        self.shape = shape
        self.description = description
        self.default = default
    
    @property
    def expression(self):
        return self._expression
    
    @expression.setter
    def expression(self, expr):
        self._expression = expr

    def setValid(self, b=True):
        if b:
            self._expression = self.default
        else:
            self._expression = None

    @property
    def valid(self):
        return self.expression is not None

    def widget(self):
        from ..widgets import ScalarFunctionWidget, VectorFunctionWidget, MatrixFunctionWidget
        shape = self._evalShape(self.shape)
        if len(shape)==0:
            widget = ScalarFunctionWidget(self.expression, valueChanged=self.__set)
        elif len(shape)==1:
            widget = VectorFunctionWidget(self.expression, valueChanged=self.__set, shape=shape)
        elif len(shape)==2:
            widget = MatrixFunctionWidget(self.expression, valueChanged=self.__set, shape=shape)
        else:
            widget = ScalarFunctionWidget(self.expression, valueChanged=self.__set)
        return widget
    
    def _evalShape(self, shape):
        res = []
        for s in shape:
            if s == "V":
                res.append(self.model.variableDimension)
            elif s == "D":
                res.append(self.fem.dimension)
            else:
                res.append(s)
        return tuple(res)

    def __set(self, val):
        self.expression = val