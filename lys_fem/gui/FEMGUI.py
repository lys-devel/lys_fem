import os
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
        self._refresh()
        self._gedit.showGeometry()
        self.closed.connect(self.__save)

    def __initProj(self):
        self._obj = FEMProject(2)
        if os.path.exists("test.dic"):
            self.__load("test.dic")

    def __initUI(self):
        self._canvas = Canvas3d()
        self._gedit = GeometryEditor(self._obj, self._canvas)
        self._medit = MeshEditor(self._obj, self._canvas)
        self._mat = TreeStyleEditor(MaterialTree(self._obj, self._canvas))
        self._model = TreeStyleEditor(ModelTree(self._obj, self._canvas))

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

    def _refresh(self):
        self._gedit.setGeometry(self._obj.geometryGenerator)
        self._mat.rootItem.setMaterials(self._obj.materials)
        self._model.rootItem.setModels(self._obj.models)

    def __save(self):
        d = self._obj.saveAsDictionary()
        with open("test.dic", "w") as f:
            f.write(str(d))

    def __load(self, file):
        with open(file, "r") as f:
            d = eval(f.read())
        self._obj.loadFromDictionary(d)

    def __test(self):
        from .. import mf
        mf.run(self._obj)
