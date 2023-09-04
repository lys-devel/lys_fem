from lys.Qt import QtCore, QtWidgets, QtGui

from .geometry import GeometryGenerator
from .geometryOrder import geometryCommands


class GeometryModel(QtCore.QAbstractItemModel):
    def __init__(self):
        super().__init__()
        self._geom = None

    def setGeometry(self, geom):
        self._geom = geom
        geom.updated.connect(self.layoutChanged.emit)

    def data(self, index, role):
        if not index.isValid() or not role == QtCore.Qt.DisplayRole or self._geom is None:
            return QtCore.QVariant()
        return self._geom.commands[index.row()].type

    def rowCount(self, parent):
        if self._geom is not None and not parent.isValid():
            return len(self._geom.commands)
        return 0

    def columnCount(self, parent):
        return 1

    def index(self, row, column, parent):
        if not parent.isValid():
            return self.createIndex(row, column, None)
        return QtCore.QModelIndex()

    def parent(self, index):
        return QtCore.QModelIndex()

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if section == 0:
                return "Model: Right click to edit"


class GeometryTree(QtWidgets.QTreeView):
    updated = QtCore.pyqtSignal()
    itemSelected = QtCore.pyqtSignal()

    def __init__(self, geom):
        super().__init__()
        self._geom = geom
        self.__model = GeometryModel()
        self.setModel(self.__model)
        self.__model.setGeometry(self._geom)
        self.selectionModel().selectionChanged.connect(self.__selectionChanged)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._buildContextMenu)

    def _buildContextMenu(self):
        menu = QtWidgets.QMenu(self)
        for item in geometryCommands:
            menu.addAction(QtWidgets.QAction(item.type, self, triggered=lambda: self.__selected(item)))
        menu.exec_(QtGui.QCursor.pos())

    def __selected(self, itemType):
        self._geom.addCommand(itemType.default)

    def selectedRow(self):
        return self.selectionModel().selectedIndexes()[0].row()

    def __selectionChanged(self):
        self.itemSelected.emit()


class GeometryEditor(QtWidgets.QWidget):
    geometryGenerated = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._geom = GeometryGenerator()
        self._geom.updated.connect(self.generate)
        self.__initlayout()

    def __initlayout(self):
        self._tree = GeometryTree(self._geom)
        self._tree.updated.connect(self.generate)
        self._tree.itemSelected.connect(self.__selectionChanged)

        self._generateAll = QtWidgets.QCheckBox("Generate all")
        self._generateAll.setChecked(True)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QPushButton("Update", clicked=self.generate))
        h.addWidget(self._generateAll)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

        self._setting = None

        self._layout = QtWidgets.QVBoxLayout()
        self._layout.addWidget(self._tree)
        self._layout.addLayout(h)
        self._layout.addWidget(line)
        self.setLayout(self._layout)

    def generate(self):
        if self._generateAll.isChecked():
            res = self._geom.generateGeometry()
        else:
            res = self._geom.generateGeometry(self._tree.selectedRow())
        self.geometryGenerated.emit(res)
        return res

    def __selectionChanged(self):
        if self._setting is not None:
            self._layout.removeWidget(self._setting)
            self._setting.deleteLater()
            self._setting = None
        self._setting = self._geom.commands[self._tree.selectedRow()].widget()
        self._layout.addWidget(self._setting)
