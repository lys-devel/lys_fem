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
    def __init__(self, parent, items=[]):
        super().__init__(items)
        self.setParent(parent)
        for item in items:
            if isinstance(item, FEMObject):
                item.setParent(self)

    def append(self, item):
        super().append(item)
        item.setParent(self)


class Coef:
    def __init__(self, expr, shape=(), description="", default=None):
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

    def setDefault(self):
        self._expression = self.default

    @property
    def valid(self):
        return self.expression is not None

    def widget(self):
        from ..widgets import ScalarFunctionWidget, VectorFunctionWidget, MatrixFunctionWidget
        if len(self.shape)==0:
            widget = ScalarFunctionWidget(None, self.expression, valueChanged=self.__set)
        elif len(self.shape)==1:
            widget = VectorFunctionWidget(None, self.expression, valueChanged=self.__set)
        elif len(self.shape)==2:
            widget = MatrixFunctionWidget(None, self.expression, valueChanged=self.__set)
        return widget

    def __set(self, val):
        self.expression = val