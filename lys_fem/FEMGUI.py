from lys.Qt import QtCore, QtWidgets
from lys.widgets import LysSubWindow

from .canvas3d import Canvas3d
from .geometryGUI import GeometryEditor
from .FEM import Elastic


class FEMGUI(LysSubWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finite-element calculation")
        self.__initUI()

    def __initUI(self):
        self._canvas = Canvas3d()

        self._gedit = GeometryEditor()
        self._gedit.geometryGenerated.connect(self._geometryChanged)
        tab = QtWidgets.QTabWidget()
        tab.addTab(self._gedit, "Geometry")

        lay = QtWidgets.QHBoxLayout()
        lay.addWidget(tab)
        lay.addWidget(self._canvas)
        lay.addWidget(QtWidgets.QPushButton("test", clicked=self.__test))
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        self.setWidget(w)
        self.adjustSize()

    def _geometryChanged(self, geom):
        self._canvas.clear()
        self._canvas.append(geom.getMeshWave(2))

    def __test(self):
        geom = self._gedit.generate()
        el = Elastic(geom)
        el.execute()
