import os
from lys.Qt import QtWidgets
from lys.widgets import LysSubWindow
from simtools import qsub

from ..fem import FEMProject
from ..widgets import TreeStyleEditor, FEMFileSystemView
from ..canvas3d import Canvas3d

from .geometryGUI import GeometryEditor
from .meshGUI import MeshEditor
from .materialGUI import MaterialTree
from .modelGUI import ModelTree
from .solverGUI import SolverTree


class FEMGUI(LysSubWindow):
    _tmpPath = "FEM/.tmp.dic"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finite-element calculation")
        self.__initProj()
        self.__initUI()
        self._gedit.showGeometry()
        self.closed.connect(self.__save)

    def __initProj(self):
        self._obj = FEMProject(2)
        if os.path.exists(self._tmpPath):
            self.__load(self._tmpPath)

    def __initUI(self):
        self._view = FEMFileSystemView()
        # self.__createContextMenus()
        input = self.__initInputUI()

        tab = QtWidgets.QTabWidget()
        tab.addTab(self._view, "Project")
        tab.addTab(input, "Calculation")

        self.setWidget(tab)
        self.adjustSize()

    def __initInputUI(self):
        self._canvas = Canvas3d()

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(QtWidgets.QPushButton("New Project", clicked=self.__new))
        buttons.addWidget(QtWidgets.QPushButton("Submit", clicked=self.__submit))

        vb = QtWidgets.QVBoxLayout()
        vb.addWidget(self._canvas)
        vb.addLayout(buttons)

        self._gedit = GeometryEditor(self._obj, self._canvas)
        self._medit = MeshEditor(self._obj, self._canvas)
        self._mat = TreeStyleEditor(MaterialTree(self._obj, self._canvas))
        self._model = TreeStyleEditor(ModelTree(self._obj, self._canvas))
        self._solver = TreeStyleEditor(SolverTree(self._obj, self._canvas))

        tab = QtWidgets.QTabWidget()
        tab.addTab(self._gedit, "Geometry")
        tab.addTab(self._medit, "Mesh")
        tab.addTab(self._mat, "Material")
        tab.addTab(self._model, "Model")
        tab.addTab(self._solver, "Solver")

        lay = QtWidgets.QHBoxLayout()
        lay.addWidget(tab)
        lay.addLayout(vb)
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        return w

    def _refresh(self):
        self._gedit.setGeometry(self._obj.geometryGenerator)
        self._mat.rootItem.setMaterials(self._obj.materials)
        self._model.rootItem.setModels(self._obj.models)
        self._solver.rootItem.setSolvers(self._obj.solvers)

    def __save(self, path=None):
        d = self._obj.saveAsDictionary()
        if path is None:
            path = self._tmpPath
        with open(path, "w") as f:
            f.write(str(d))

    def __load(self, file):
        with open(file, "r") as f:
            d = eval(f.read())
        self._obj.loadFromDictionary(d)

    def __new(self):
        dim, ok = QtWidgets.QInputDialog.getInt(self, "Dimension", "Enter the dimension of simulation.", value=3, min=1, max=3)
        if ok:
            self._obj = FEMProject(dim)
            self._refresh()

    def __submit(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "Submit", "Enter the name.", text="test")
        if ok:
            os.makedirs("FEM/" + text, exist_ok=True)
            self.__save(path="FEM/" + text + "/input.dic")
            qsub.execute("python -m lys_fem.mf --input input.dic --mpi=True", "FEM/" + text, ncore=4)
