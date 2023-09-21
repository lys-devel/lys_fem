from lys.Qt import QtCore, QtWidgets


class MeshEditor(QtWidgets.QWidget):
    def __init__(self, obj, canvas):
        super().__init__()
        self._obj = obj
        self._canvas = canvas
        self._mesher = obj.mesher
        self.__initlayout()

    def __initlayout(self):
        self._refine = QtWidgets.QSpinBox()
        self._refine.setRange(0, 5)
        self._refine.valueChanged.connect(self._mesher.setRefinement)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._refine)
        layout.addWidget(QtWidgets.QPushButton("Generate Mesh", clicked=self._showMesh))
        layout.addStretch()
        self.setLayout(layout)

    def _showMesh(self):
        mesh = self._obj.getMeshWave()
        with self._canvas.delayUpdate():
            self._canvas.clear()
            obj = self._canvas.append(mesh)
            for o in obj:
                o.showEdges(True)
