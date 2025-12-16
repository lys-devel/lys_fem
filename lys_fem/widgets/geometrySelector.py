import weakref
from lys.Qt import QtCore, QtWidgets, QtGui


class GeometrySelector(QtWidgets.QWidget):
    def __init__(self, canvas, fem, selected, acceptedTypes=["All", "Selected", "Group"], autoStart=True):
        super().__init__()
        dimDict = {"Domain": fem.dimension, "Boundary": fem.dimension-1, "Volume": 3, "Surface": 2, "Edge": 1, "Point": 0}
        self._dim = dimDict[selected.geometryType]
        self._geomType = selected.geometryType
        self._selected = selected
        self._canvas = canvas
        self._fem = fem
        self.__initlayout(selected.geometryType, selected, acceptedTypes, dimDict)
        if autoStart:
            if selected.selectionType() == "all":
                self.__showGeometry()
            else:
                self.startSelection()

    def __initlayout(self, title, selected, acceptedTypes, dimDict):
        self._type = QtWidgets.QComboBox()
        self._type.addItems(acceptedTypes)
        self._type.setCurrentText(selected.selectionType())
        self._type.currentTextChanged.connect(self.__typeChanged)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Type"))
        h.addWidget(self._type)

        self._model = _SelectionModel(selected, title)
        self._tree = QtWidgets.QTreeView()
        self._tree.setModel(self._model)
        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._buildContextMenu)
        self._selectBtn = QtWidgets.QPushButton("Start Selection", clicked=self.__start)
        self._selectBtn.setCheckable(True)

        self._groups = QtWidgets.QTreeWidget()
        self._groups.setHeaderLabels(["Group"])
        for key, grp in self._fem.geometries.groups.items():
            if dimDict[grp.geometryType] == self._dim:
                item = QtWidgets.QTreeWidgetItem(self._groups)
                item.setText(0, key)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(0, QtCore.Qt.Checked if key in selected.getSelection() else QtCore.Qt.Unchecked)
        self._groups.itemChanged.connect(self._groupChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(h)
        layout.addWidget(self._tree)
        layout.addWidget(self._selectBtn)
        layout.addWidget(self._groups)
        self.setLayout(layout)
        self.__setEnabled()

    def _buildContextMenu(self):
        menu = QtWidgets.QMenu()
        menu.addAction(QtWidgets.QAction("Remove", self, triggered=self.__remove))
        menu.addAction(QtWidgets.QAction("Clear", self, triggered=self.__clear))
        menu.exec_(QtGui.QCursor.pos())

    def __remove(self):
        item = self._selected[self._tree.currentIndex().row()]
        self._selected.remove(item)
        self.__showGeometry()
        self._model.layoutChanged.emit()

    def __clear(self):
        self._selected.clear()
        self.__showGeometry()
        self._model.layoutChanged.emit()

    def __typeChanged(self, text):
        if text == "All":
            self._selected.setSelection("all")
            self.__showGeometry()
        elif text=="Selected":
            self._selected.setSelection([])
            self.startSelection()
        else:
            self._selected.setSelection([])
            self.__showGeometry()
        self._model.layoutChanged.emit()
        self.__setEnabled()

    def _groupChanged(self, item, column):
        if item.checkState(column):
            self._selected.append(item.text(column))
        else:
            self._selected.remove(item.text(column))
        self.__showGeometry()

    def __setEnabled(self):
        if self._type.currentText() == "All":
            self._tree.hide()
            self._selectBtn.hide()
            self._groups.hide()
        elif self._type.currentText() == "Selected":
            self._tree.show()
            self._selectBtn.show()
            self._groups.hide()
        else:
            self._tree.hide()
            self._selectBtn.hide()
            self._groups.show()

    def __showGeometry(self):
        mesh = self._fem.getMeshWave(self._dim, nomesh=True)
        with self._canvas.delayUpdate():
            self._canvas.clear()
            objs = self._canvas.append(mesh)
            for obj, m in zip(objs, mesh):
                self.__setColor(obj, self._selected.check(m.note["tag"]))
            if self._geomType == "Surface" and self._dim != 2:
                mesh = self._fem.getMeshWave(self._dim + 1, nomesh=True)
                objs = self._canvas.append(mesh)
                for obj, m in zip(objs, mesh):
                    self.__setColor(obj, False)

    def startSelection(self):
        self._selectBtn.setChecked(True)
        self.__start()

    def __start(self):
        if self._selectBtn.isChecked():
            types = {0: "point", 1: "line", 2: "surface", 3: "volume"}
            self.__showGeometry()
            self._canvas.startPicker(types[self._dim])
            self._canvas.objectPicked.connect(self.__picked)
            self._canvas.pickingFinished.connect(self.__finishSelection)
        else:
            self._canvas.endPicker()

    def __finishSelection(self):
        self._canvas.objectPicked.disconnect(self.__picked)
        self._canvas.pickingFinished.disconnect(self.__finishSelection)
        self._selectBtn.setChecked(False)

    def __picked(self, item):
        tag = item.getWave().note["tag"]
        if self._selected.check(tag):
            self._selected.remove(tag)
            self.__setColor(item, False)
        else:
            self._selected.append(tag)
            self.__setColor(item, True)
        self._model.layoutChanged.emit()

    def __setColor(self, obj, selected):
        if selected:
            color = "#adff2f"
        else:
            color = "#cccccc"
        obj.setColor(color, type="color")


class _SelectionModel(QtCore.QAbstractItemModel):
    def __init__(self, parent, title):
        super().__init__()
        self._title = title
        self._parent = weakref.ref(parent)

    def data(self, index, role):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return self.selection.getSelection()[index.row()]
        return QtCore.QVariant()

    def rowCount(self, parent):
        if parent.isValid():
            return 0
        else:
            return len(self.selection.getSelection())

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
                return self._title

    @property
    def selection(self):
        return self._parent()
