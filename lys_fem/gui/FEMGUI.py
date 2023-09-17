from lys.Qt import QtCore, QtWidgets
from lys.widgets import LysSubWindow

from ..fem import FEMProject, OccMesher
from ..widgets import TreeStyleEditor
from ..canvas3d import Canvas3d

from .geometryGUI import GeometryEditor
from .meshGUI import MeshEditor
from .materialGUI import MaterialTree
from .modelGUI import ModelTree


class FEMGUI(LysSubWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finite-element calculation")
        self.__initProj()
        self.__initUI()
        self._showGeometry()
        self.closed.connect(self.__save)

    def __initProj(self):
        self._obj = FEMProject(2)
        self.__load("test.dic")

    def __initUI(self):
        self._canvas = Canvas3d()

        self._gedit = GeometryEditor(self._obj.geometryGenerator)
        self._gedit.showGeometry.connect(self._showGeometry)
        self._medit = MeshEditor(self._obj.mesher)
        self._medit.showMesh.connect(self._showMesh)
        self._mat = TreeStyleEditor(MaterialTree(self._obj, self._canvas))
        self._model = TreeStyleEditor(ModelTree(self._obj._models))
        tab = QtWidgets.QTabWidget()
        tab.addTab(self._gedit, "Geometry")
        tab.addTab(self._medit, "Mesh")
        tab.addTab(self._mat, "Material")
        tab.addTab(self._model, "Model")

        lay = QtWidgets.QHBoxLayout()
        lay.addWidget(tab)
        lay.addWidget(self._canvas)
        lay.addWidget(QtWidgets.QPushButton("test", clicked=self.__test))
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        self.setWidget(w)
        self.adjustSize()

    def __save(self):
        d = self._obj.saveAsDictionary()
        with open("test.dic", "w") as f:
            f.write(str(d))

    def __load(self, file):
        with open(file, "r") as f:
            d = eval(f.read())
        self._obj.loadFromDictionary(d)

    def _showGeometry(self):
        geom = self._gedit.generate()
        with self._canvas.delayUpdate():
            self._canvas.clear()
            self._canvas.append(OccMesher().getMeshWave(geom, dim=self._obj.dimension))

    def _showMesh(self):
        mesh = self._obj.getMeshWave()
        with self._canvas.delayUpdate():
            self._canvas.clear()
            obj = self._canvas.append(mesh)
            for o in obj:
                o.showEdges(True)

    def __test(self):
        self._obj.geometryGenerator.generateGeometry().export("test.msh", self._obj.dimension)
