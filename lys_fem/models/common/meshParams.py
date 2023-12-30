import numpy as np
import sympy as sp

from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import VectorFunctionWidget


class InfiniteVolumeParams(FEMParameter):
    def __init__(self, abc=[1,1,1], ABC=[2,2,2], alpha=1, domain="x+"):
        self.abc = abc
        self.ABC = ABC
        self.domain = domain
        self.alpha = alpha

    @classmethod
    @property
    def name(cls):
        return "Infinite Volume (3D)"

    def getParameters(self, dim):
        X, Y, Z = self._constructJ()
        x,y,z = sp.symbols("x,y,z", real=True)
        J = [[X.diff(x), X.diff(y), X.diff(z)],[Y.diff(x), Y.diff(y), Y.diff(z)],[Z.diff(x), Z.diff(y), Z.diff(z)]]
        return {"CoordsTransform": [X,Y,Z], "J": J}

    def _constructJ(self):
        a,b,c = self.abc
        A,B,C = self.ABC
        alpha = self.alpha

        Cx = np.array([0, b-a*(B-b)/(A-a), c-a*(C-c)/(A-a)])
        Cy = np.array([a-b*(A-a)/(B-b), 0, c-b*(C-c)/(B-b)])
        Cz = np.array([a-c*(A-a)/(C-c), b-c*(B-b)/(C-c), 0])
        Cb = (Cx + Cy + Cz)/3

        x,y,z = sp.symbols("x,y,z", real=True)
        if self.domain == "x+":
            X = A-(A-a)*((a-Cb[0])/(x-Cb[0]))**alpha 
            Y = y * (X-Cy[0]) / (x-Cy[0])
            Z = z * (X-Cz[0]) / (x-Cz[0])
        if self.domain == "x-":
            X = -(A-(A-a)*((a-Cb[0])/(-x-Cb[0]))**alpha) 
            Y = y * (-X-Cy[0]) / (-x-Cy[0])
            Z = z * (-X-Cz[0]) / (-x-Cz[0])
        elif self.domain == "y+":
            Y = B-(B-b)*((b-Cb[1])/(y-Cb[1]))**alpha
            X = x * (Y-Cx[1]) / (y-Cx[1])
            Z = z * (Y-Cz[1]) / (y-Cz[1])
        elif self.domain == "y-":
            Y = -(B-(B-b)*((b-Cb[1])/(-y-Cb[1]))**alpha)
            X = x * (-Y-Cx[1]) / (-y-Cx[1])
            Z = z * (-Y-Cz[1]) / (-y-Cz[1])
        elif self.domain == "z+":
            Z = C-(C-c)*((c-Cb[2])/(z-Cb[2]))**alpha
            X = x * (Z -Cx[2]) / (z-Cx[2])
            Y = y * (Z -Cy[2]) / (z-Cy[2])
        elif self.domain == "z-":
            Z = -(C-(C-c)*((c-Cb[2])/(-z-Cb[2]))**alpha)
            X = x * (-Z -Cx[2]) / (-z-Cx[2])
            Y = y * (-Z -Cy[2]) / (-z-Cy[2])
        return [X,Y,Z]

    def widget(self):
        return _InfiniteVolumeWidget(self)


class InfinitePlaneParams(FEMParameter):
    def __init__(self, ab=[1,1], AB=[2,2], alpha=1, domain="x+"):
        self.ab = ab
        self.AB = AB
        self.domain = domain
        self.alpha = alpha

    @classmethod
    @property
    def name(cls):
        return "Infinite Plane (2D)"

    def getParameters(self, dim):
        X, Y = self._constructJ()
        return {"CoordsTransform": [X,Y]}

    def _constructJ(self):
        a,b = self.ab
        A,B = self.AB
        alpha = self.alpha

        Cx = np.array([0, b-a*(B-b)/(A-a)])
        Cy = np.array([a-b*(A-a)/(B-b), 0])
        Cb = (Cx + Cy)/2

        x,y = sp.symbols("x,y", real=True)
        if self.domain == "x+":
            X = A-(A-a)*((a-Cb[0])/(x-Cb[0]))**alpha 
            Y = y * (X-Cy[0]) / (x-Cy[0])
        if self.domain == "x-":
            X = -(A-(A-a)*((a-Cb[0])/(-x-Cb[0]))**alpha) 
            Y = y * (-X-Cy[0]) / (-x-Cy[0])
        elif self.domain == "y+":
            Y = B-(B-b)*((b-Cb[1])/(y-Cb[1]))**alpha
            X = x * (Y-Cx[1]) / (y-Cx[1])
        elif self.domain == "y-":
            Y = -(B-(B-b)*((b-Cb[1])/(-y-Cb[1]))**alpha)
            X = x * (-Y-Cx[1]) / (-y-Cx[1])
        return [X,Y]

    def widget(self):
        return _InfinitePlaneWidget(self)


class _InfiniteVolumeWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self.__initlayout()

    def __initlayout(self):
        self._domain = QtWidgets.QComboBox()
        self._domain.addItems(["x+", "x-", "y+", "y-", "z+", "z-"])
        self._domain.setCurrentText(self._param.domain)
        self._domain.currentTextChanged.connect(self.__set)
        self._p1 = VectorFunctionWidget("Point1", self._param.abc, valueChanged=self.__set)
        self._p2 = VectorFunctionWidget("Point2", self._param.ABC, valueChanged=self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._domain)
        layout.addWidget(self._p1)
        layout.addWidget(self._p2)
        self.setLayout(layout)

    def __set(self):
        self._param.domain = self._domain.currentText()
        self._param.abc = self._p1.value()
        self._param.ABC = self._p2.value()


class _InfinitePlaneWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self.__initlayout()

    def __initlayout(self):
        self._domain = QtWidgets.QComboBox()
        self._domain.addItems(["x+", "x-", "y+", "y-"])
        self._domain.setCurrentText(self._param.domain)
        self._domain.currentTextChanged.connect(self.__set)
        self._p1 = VectorFunctionWidget("Point1", self._param.ab, valueChanged=self.__set)
        self._p2 = VectorFunctionWidget("Point2", self._param.AB, valueChanged=self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._domain)
        layout.addWidget(self._p1)
        layout.addWidget(self._p2)
        self.setLayout(layout)

    def __set(self):
        self._param.domain = self._domain.currentText()
        self._param.ab = self._p1.value()
        self._param.AB = self._p2.value()


