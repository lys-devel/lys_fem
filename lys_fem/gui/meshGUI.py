from lys.Qt import QtCore, QtWidgets


class MeshEditor(QtWidgets.QWidget):
    showMesh = QtCore.pyqtSignal()

    def __init__(self, mesher):
        super().__init__()
        self._mesher = mesher
        self.__initlayout()

    def __initlayout(self):
        self._refine = QtWidgets.QSpinBox()
        self._refine.setRange(0, 5)
        self._refine.valueChanged.connect(self._mesher.setRefinement)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._refine)
        layout.addWidget(QtWidgets.QPushButton("Generate Mesh", clicked=self.showMesh.emit))
        layout.addStretch()
        self.setLayout(layout)
