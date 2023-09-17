import weakref
from lys.Qt import QtCore, QtWidgets, QtGui


class GeometrySelector(QtWidgets.QWidget):
    selectionChanged = QtCore.pyqtSignal(list)

    def __init__(self, title, dim, canvas, fem, selected=[]):
        super().__init__()
        self._dim = dim
        self._selected = list(selected)
        self._canvas = canvas
        self._fem = fem
        self.__initlayout(title)

    def __initlayout(self, title):
        self._model = _SelectionModel(self, title)
        self._tree = QtWidgets.QTreeView()
        self._tree.setModel(self._model)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(QtWidgets.QPushButton("Start Selection", clicked=self.__startSelection))

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)
        layout.addLayout(buttons)
        self.setLayout(layout)

    @property
    def selectedItems(self):
        return self._selected

    def __startSelection(self):
        types = {1: "line", 2: "surface", 3: "volume"}
        mesh = self._fem.getMeshWave(self._dim)
        self._canvas.clear()
        objs = self._canvas.append(mesh)
        for obj, m in zip(objs, mesh):
            if m.note["tag"] in self._selected:
                obj.setColor("#ff0000", type="color")
            else:
                obj.setColor('#333333', type="color")
        self._canvas.startPicker(types[self._dim])
        self._canvas.objectPicked.connect(self.__picked)

    def __picked(self, item):
        tag = item.getWave().note["tag"]
        if tag in self._selected:
            self._selected.remove(tag)
            item.setColor('#333333', type="color")
        else:
            self._selected.append(tag)
            item.setColor('#ff0000', type="color")
        self._model.layoutChanged.emit()
        self.selectionChanged.emit(self._selected)


class _SelectionModel(QtCore.QAbstractItemModel):
    def __init__(self, parent, title):
        super().__init__()
        self._title = title
        self._parent = weakref.ref(parent)

    def data(self, index, role):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return self.selector.selectedItems[index.row()]
        return QtCore.QVariant()

    def rowCount(self, parent):
        if parent.isValid():
            return 0
        else:
            return len(self.selector.selectedItems)

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
    def selector(self):
        return self._parent()
