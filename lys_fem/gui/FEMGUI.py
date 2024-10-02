import os
from lys.Qt import QtWidgets
from lys.widgets import LysSubWindow, lysCanvas3D
from simtools import qsub

from ..fem import FEMProject
from ..widgets import TreeStyleEditor, FEMFileSystemView, FEMFileDialog

from .geometryGUI import GeometryEditor
from .meshGUI import MeshEditor
from .parameterGUI import VariableTree
from .modelGUI import ModelTree
from .solverGUI import SolverTree
from .solutionGUI import SolutionTree


class FEMGUI(LysSubWindow):
    _tmpPath = "FEM/.tmp.dic"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finite-element calculation")
        self.__initProj()
        self.__initUI()
        self._gedit.showGeometry()
        self.closed.connect(lambda x: self.__save())

    def __initProj(self):
        self._obj = FEMProject(2)
        if os.path.exists(self._tmpPath):
            self.__load(self._tmpPath)

    def __initUI(self):
        input = self.__initInputUI()
        output = self.__initOutputUI()

        tab = QtWidgets.QTabWidget()
        tab.addTab(input, "Calculation")
        tab.addTab(output, "Results")

        self.setWidget(tab)
        self.adjustSize()

    def __initInputUI(self):
        self._canvas = lysCanvas3D()

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(QtWidgets.QPushButton("New Project", clicked=self.__new))
        buttons.addWidget(QtWidgets.QPushButton("Load", clicked=self.__loadProj))
        buttons.addWidget(QtWidgets.QPushButton("Submit", clicked=self.__submit))

        vb = QtWidgets.QVBoxLayout()
        vb.addWidget(self._canvas)
        vb.addLayout(buttons)

        self._gedit = GeometryEditor(self._obj, self._canvas)
        self._medit = MeshEditor(self._obj, self._canvas)
        self._mat = TreeStyleEditor(VariableTree(self._obj, self._canvas))
        self._model = TreeStyleEditor(ModelTree(self._obj, self._canvas))
        self._solver = TreeStyleEditor(SolverTree(self._obj, self._canvas))

        tab = QtWidgets.QTabWidget()
        tab.addTab(self._gedit, "Geometry")
        tab.addTab(self._medit, "Mesh")
        tab.addTab(self._mat, "Params")
        tab.addTab(self._model, "Model")
        tab.addTab(self._solver, "Solver")

        lay = QtWidgets.QHBoxLayout()
        lay.addWidget(tab)
        lay.addLayout(vb)
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        return w

    def __initOutputUI(self):
        self._canvas_out = lysCanvas3D()
        self._tree = SolutionTree(self._canvas_out)
        self._edit = TreeStyleEditor(self._tree)
        self._view = FEMFileSystemView()
        self._view.selectionChanged.connect(lambda: self._tree.setSolutionPath(self._view.getCurrentPath()))

        lay = QtWidgets.QHBoxLayout()
        lay.addWidget(self._view)
        lay.addWidget(self._edit)
        lay.addWidget(self._canvas_out)
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        return w

    def _refresh(self):
        self._gedit.setGeometry(self._obj.geometries)
        self._medit.setMesher(self._obj.mesher)
        self._mat.rootItem.setVariables(self._obj.materials)
        self._model.rootItem.setModels(self._obj.models)
        self._solver.rootItem.setSolvers(self._obj.solvers)

    def __save(self, path=None, parallel=False):
        d = self._obj.saveAsDictionary(parallel=parallel)
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
            self._obj.reset(dim)
            self._refresh()

    def __submit(self):
        d = _SubmitDialog(self, self._obj.submitSetting)
        if d.exec_():
            sub = d.getSettingDict()
            self._obj.submitSetting.update(sub)
            path = d.getPath()
            os.makedirs(path, exist_ok=True)
            self.__save(path=path + "/input.dic", parallel=sub["type"] != "Serial")
            ncore = 1 if sub["type"] == "Serial" else sub["ncore"]
            if sub["type"] in ["Serial", "Parallel"]:
                qsub.execute("python -m lys_fem.ngs", path, ncore=ncore)
            elif sub["type"] == "qsub":
                command = "python -m lys_fem.ngs"
                qsub.submit(command, path=path, ncore=ncore, nodes=sub["nnodes"], name="mfem", queue=sub["queue"])

    def __loadProj(self):
        d = FEMFileDialog(self)
        ok = d.exec_()
        if ok:
            path = "FEM/" + d.result + "/input.dic"
            self.__load(path)
            self._refresh()


class _SubmitDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, setting=None):
        super().__init__(parent)
        self.__initLayout()
        if setting is not None:
            self.setSettingDict(setting)

    def __initLayout(self):
        self._view = FEMFileSystemView(filter=False)
        self._name = QtWidgets.QLineEdit()
        self._type = QtWidgets.QComboBox()
        self._type.addItems(["Serial", "Parallel", "qsub"])
        self._np = QtWidgets.QSpinBox()
        self._np.setRange(1, 100000000)
        self._np.setEnabled(False)
        self._nodes = QtWidgets.QSpinBox()
        self._nodes.setRange(1, 100000000)
        self._nodes.setEnabled(False)
        self._queue = QtWidgets.QLineEdit()
        self._queue.setEnabled(False)
        self._type.currentTextChanged.connect(self.__changeType)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Name"), 0, 0)
        grid.addWidget(self._name, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Type"), 1, 0)
        grid.addWidget(self._type, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Number of cores"), 2, 0)
        grid.addWidget(self._np, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Number of nodes"), 3, 0)
        grid.addWidget(self._nodes, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Queue name"), 4, 0)
        grid.addWidget(self._queue, 4, 1)

        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QPushButton('O K', clicked=self.accept))
        h1.addWidget(QtWidgets.QPushButton('CALCEL', clicked=self.reject))

        v1 = QtWidgets.QVBoxLayout()
        v1.addLayout(grid)
        v1.addStretch()
        v1.addLayout(h1)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._view)
        layout.addLayout(v1)

        self.setLayout(layout)

    def __changeType(self, type):
        if type == "Serial":
            self._np.setEnabled(False)
            self._nodes.setEnabled(False)
            self._queue.setEnabled(False)
        elif type == "Parallel":
            self._np.setEnabled(True)
            self._nodes.setEnabled(False)
            self._queue.setEnabled(False)
        elif type == "qsub":
            self._np.setEnabled(True)
            self._nodes.setEnabled(True)
            self._queue.setEnabled(True)

    def getPath(self):
        return "FEM/" + self._view.getCurrentPath() + "/" + self._name.text()

    def getSettingDict(self):
        return {"name": self._name.text(), "type": self._type.currentText(), "ncore": self._np.value(), "nnodes": self._nodes.value(), "queue": self._queue.text()}

    def setSettingDict(self, d):
        if "name" in d:
            self._name.setText(d["name"])
        if "type" in d:
            self._type.setCurrentText(d["type"])
        if "ncore" in d:
            self._np.setValue(d["ncore"])
        if "nnodes" in d:
            self._nodes.setValue(d["nnodes"])
        if "queue" in d:
            self._queue.setText(d["queue"])
