import numpy as np
import sympy as sp

from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import VectorFunctionWidget


class InfiniteVolumeParams(FEMParameter):
    def __init__(self, abc=[1,1,1], ABC=[2,2,2], alpha=2, domain="x+"):
        self.abc = abc
        self.ABC = ABC
        self.domain = domain
        self.alpha = alpha

    @classmethod
    @property
    def name(cls):
        return "Infinite Volume (3D)"

    def getParameters(self, dim):
        return {"CoordsTransform": self._constructJ()}

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
            Z = C-(C-c)*((C-Cb[2])/(z-Cb[2]))**alpha
            X = x * (Z -Cx[2]) / (z-Cx[2])
            Y = y * (Z -Cy[2]) / (z-Cy[2])
        elif self.domain == "z-":
            Z = -(C-(C-c)*((C-Cb[2])/(-z-Cb[2]))**alpha)
            X = x * (-Z -Cx[2]) / (-z-Cx[2])
            Y = y * (-Z -Cy[2]) / (-z-Cy[2])
        return [X,Y,Z]

    def widget(self):
        return _InfiniteVolumeWidget(self)


class _InfiniteVolumeWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self.__initlayout()

    def __initlayout(self):
        self._p1 = VectorFunctionWidget("Point1", self._param.abc, valueChanged=self.__set)
        self._p2 = VectorFunctionWidget("Point2", self._param.ABC, valueChanged=self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._p1)
        layout.addWidget(self._p2)
        self.setLayout(layout)

    def __set(self):
        self._param.abc = self._p1.value()
        self._param.ABC = self._p2.value()
