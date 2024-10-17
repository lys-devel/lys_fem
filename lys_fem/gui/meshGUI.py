from lys.Qt import QtWidgets

from ..fem import GeometrySelection
from ..widgets import FEMTreeItem, GeometrySelector, TreeStyleEditor


class MeshEditor(QtWidgets.QWidget):
    def __init__(self, obj, canvas):
        super().__init__()
        self._obj = obj
        self._canvas = canvas
        self._mesher = obj.mesher
        self.__initlayout()

    def __initlayout(self):
        self._mt = MeshTree(self._obj, self._canvas)
        self._tr = TreeStyleEditor(self._mt)

        self._tr.layout.insertWidget(1, QtWidgets.QPushButton("Generate Mesh", clicked=self._showMesh))

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tr)
        self.setLayout(layout)

    def _showMesh(self):
        mesh = self._obj.getMeshWave()
        elements, nodes = 0, 0
        for m in mesh:
            elements += sum([len(value) for value in m.note["elements"].values()])
            nodes += len(m.note["nodes"])
        print("Mesh:", elements, "elements,", nodes, "nodes")
        with self._canvas.delayUpdate():
            self._canvas.clear()
            obj = self._canvas.append(mesh)
            for o in obj:
                o.setColor("#cccccc", type="color")
                o.showMeshes(True)

    def setMesher(self, mesher):
        self._mesher = mesher
        self._mt.setMesher(mesher)


class MeshTree(FEMTreeItem):
    def __init__(self, obj, canvas):
        super().__init__(fem=obj, canvas=canvas)
        self.setMesher(obj.mesher)

    def setMesher(self, mesher):
        self.clear()
        self._mesher = mesher
        super().append(_RefineGUI(self._mesher, self))
        super().append(_TransfiniteGroupGUI(self._mesher, self))
        super().append(_PeriodicPairsGUI(self._mesher, self))


class _RefineGUI(FEMTreeItem):
    def __init__(self, mesher, parent):
        super().__init__(parent, children=[_PartialRefineGUI(p, self) for p in mesher.partialRefinement])
        self._mesher = mesher

    @property
    def name(self):
        return "Refinement"

    @property
    def widget(self):
        return _RefineWidget(self.canvas(), self.fem(), self._mesher)

    def _append(self, type):
        geom = self._mesher.addPartialRefinement(type)
        super().append(_PartialRefineGUI(geom, self))

    def remove(self, init):
        i = super().remove(init)
        self._mesher.partialRefinement.remove(self._mesher.partialRefinement[i])

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Add Volume Refinement", self.treeWidget(), triggered=lambda: self._append("Volume")))
        self._menu.addAction(QtWidgets.QAction("Add Surface Refinement", self.treeWidget(), triggered=lambda: self._append("Surface")))
        self._menu.addAction(QtWidgets.QAction("Add Edge Refinement", self.treeWidget(), triggered=lambda: self._append("Edge")))
        return self._menu


class _PartialRefineGUI(FEMTreeItem):
    def __init__(self, geom, parent):
        super().__init__(parent)
        self._geom = geom

    @property
    def name(self):
        return self._geom.geometryType+" Refinement"

    @property
    def widget(self):
        return _PartialRefineWidget(self.canvas(), self.fem(), self._geom)

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class _RefineWidget(QtWidgets.QWidget):
    def __init__(self, canvas, fem, mesher):
        super().__init__()
        self._mesher = mesher
        self.__initlayout(canvas, fem, mesher)

    def __initlayout(self, canvas, fem, mesher):
        self._refine = QtWidgets.QSpinBox()
        self._refine.setRange(0, 9)
        self._refine.setValue(self._mesher.refinement)
        self._refine.valueChanged.connect(self._mesher.setRefinement)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Refinement factor"))
        layout.addWidget(self._refine)
        self.setLayout(layout)


class _PartialRefineWidget(QtWidgets.QWidget):
    def __init__(self, canvas, fem, geom):
        super().__init__()
        self._geom = geom
        self.__initlayout(canvas, fem, geom)

    def __initlayout(self, canvas, fem, geom):
        self._refine = QtWidgets.QSpinBox()
        self._refine.setRange(1, 100000000)
        self._refine.setValue(geom.factor)
        self._refine.valueChanged.connect(self._setFactor)

        self._sel = GeometrySelector(canvas, fem, geom)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Refinement factor"))
        layout.addWidget(self._refine)
        layout.addWidget(self._sel)
        self.setLayout(layout)

    def _setFactor(self, v):
        self._geom.factor=v


class _TransfiniteGroupGUI(FEMTreeItem):
    def __init__(self, mesher, parent):
        super().__init__(parent, children=[_TransfiniteGUI(t, self) for t in mesher.transfinite])
        self._mesher = mesher

    @property
    def name(self):
        return "Transfinite"

    def _append(self, type):
        geom = self._mesher.addTransfinite(type)
        super().append(_TransfiniteGUI(geom, self))

    def remove(self, init):
        i = super().remove(init)
        self._mesher.transfinite.remove(self._mesher.transfinite[i])

    @ property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Add Transfinite Volume", self.treeWidget(), triggered=lambda: self._append("Volume")))
        self._menu.addAction(QtWidgets.QAction("Add Transfinite Surface", self.treeWidget(), triggered=lambda: self._append("Surface")))
        return self._menu


class _TransfiniteGUI(FEMTreeItem):
    def __init__(self, geom, parent):
        super().__init__(parent)
        self._geom = geom

    @property
    def name(self):
        return "Transfinite " + self._geom.geometryType

    @property
    def widget(self):
        return _TransfiniteWidget(self.canvas(), self.fem(), self._geom)

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class _TransfiniteWidget(QtWidgets.QWidget):
    def __init__(self, canvas, fem, geom):
        super().__init__()
        self._geom = geom
        self.__initlayout(canvas, fem, geom)

    def __initlayout(self, canvas, fem, geom):
        self._sel = GeometrySelector(canvas, fem, geom)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._sel)
        self.setLayout(layout)


class _PeriodicPairsGUI(FEMTreeItem):
    def __init__(self, mesher, parent):
        super().__init__(parent, children=[_PeriodicGUI(p, self) for p in mesher.periodicPairs])
        self._mesher = mesher

    @property
    def name(self):
        return "Periodicity"

    def append(self):
        p = (GeometrySelection(geometryType="Surface"), GeometrySelection(geometryType="Surface"))
        self._mesher.periodicPairs.append(p)
        super().append(_PeriodicGUI(p, self))

    def remove(self, item):
        i = super().remove(item)
        self._mesher.periodicPairs.remove(self._mesher.periodicPairs[i])

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Add periodicity", self.treeWidget(), triggered=self.append))
        return self._menu


class _PeriodicGUI(FEMTreeItem):
    def __init__(self, pair, parent):
        super().__init__(parent)
        self._pair = pair

    @property
    def name(self):
        return "Periodic Pair"

    @property
    def widget(self):
        return _PeriodicWidget(self.canvas(), self.fem(), self._pair)

    @property
    def menu(self):
        self._menu = QtWidgets.QMenu()
        self._menu.addAction(QtWidgets.QAction("Remove", self.treeWidget(), triggered=lambda: self.parent.remove(self)))
        return self._menu


class _PeriodicWidget(QtWidgets.QWidget):
    def __init__(self, canvas, fem, pair):
        super().__init__()
        self._pair = pair
        self.__initlayout(canvas, fem, pair)

    def __initlayout(self, canvas, fem, pair):
        self._src = GeometrySelector(canvas, fem, pair[0], acceptedTypes=["Selected"])
        self._dst = GeometrySelector(canvas, fem, pair[1], acceptedTypes=["Selected"], autoStart=False)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Source surface"))
        layout.addWidget(self._src)
        layout.addWidget(QtWidgets.QLabel("Dest surface"))
        layout.addWidget(self._dst)
        self.setLayout(layout)
