from lys.Qt import QtCore, QtWidgets, QtGui
import weakref


class TreeStyleEditor(QtWidgets.QWidget):
    def __init__(self, tree):
        super().__init__()
        self._root = tree
        self._treeModel = _TreeModel(tree)
        self.__initlayout(tree)
        tree._setTreeWidget(self._tree, self._treeModel)

    def __initlayout(self, tree):
        self._tree = QtWidgets.QTreeView()
        self._tree.setModel(self._treeModel)
        self._tree._treeModel = self._treeModel
        self._tree.selectionModel().selectionChanged.connect(self.__selectionChanged)
        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._buildContextMenu)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

        self._setting = tree.widget

        self._layout = QtWidgets.QVBoxLayout()
        self._layout.addWidget(self._tree)
        self._layout.addWidget(line)
        self._layout.addWidget(self._setting)
        self.setLayout(self._layout)

    def _buildContextMenu(self):
        indexes = self._tree.selectedIndexes()
        if len(indexes) == 0:
            menu = self._root.menu
        else:
            menu = indexes[0].internalPointer().menu
        if menu is not None:
            menu.exec_(QtGui.QCursor.pos())

    def __selectionChanged(self):
        if self._setting is not None:
            self._layout.removeWidget(self._setting)
            self._setting.deleteLater()
            self._setting = None
        indexes = self._tree.selectedIndexes()
        if len(indexes) == 0:
            return self._root.widget
        else:
            self._setting = indexes[0].internalPointer().widget
        self._layout.addWidget(self._setting)

    def update(self):
        self._treeModel.layoutChanged.emit()

    @property
    def treeView(self):
        return self._tree

    @property
    def layout(self):
        return self._layout


class _TreeModel(QtCore.QAbstractItemModel):
    def __init__(self, root):
        super().__init__()
        self._root = root

    def data(self, index, role):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return index.internalPointer().name
        return QtCore.QVariant()

    def rowCount(self, parent):
        if parent.isValid():
            return len(parent.internalPointer().children)
        else:
            return len(self._root.children)

    def columnCount(self, parent):
        return 1

    def index(self, row, column, parent):
        if parent.isValid():
            par = parent.internalPointer()
        else:
            par = self._root
        if row < len(par.children):
            return self.createIndex(row, column, par.children[row])
        else:
            return QtCore.QModelIndex()

    def parent(self, index):
        if index.isValid():
            par = index.internalPointer().parent
            if par == self._root:
                return QtCore.QModelIndex()
            else:
                return self.createIndex(par.parent.children.index(par), 0, par)
        return QtCore.QModelIndex()

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if section == 0:
                return "Model: Right click to edit"


class TreeItem(object):
    def __init__(self, parent=None):
        super().__init__()
        self._parent = None
        if parent is not None:
            self._parent = weakref.ref(parent)
        self._treeWidget = None
        self._treeModel = None

    @property
    def name(self):
        return "Not defined"

    @property
    def parent(self):
        if self._parent is None:
            return None
        else:
            return self._parent()

    @property
    def children(self):
        return []

    @property
    def widget(self):
        return QtWidgets.QLabel("Click tree item to edit")

    @property
    def currentItem(self):
        self.treeWidget.currentIndex().internalPointer()

    @property
    def menu(self):
        return None

    def _setTreeWidget(self, tree, model):
        self._treeWidget = weakref.ref(tree)
        self._treeModel = weakref.ref(model)

    def treeWidget(self):
        if self._treeWidget is None:
            return self.parent.treeWidget()
        else:
            return self._treeWidget()

    def _treeModel(self):
        if self._treeModel is None:
            return self.parent.treeModel()
        else:
            return self._treeModel()

    def _treeIndex(self):
        if self.parent is None:
            return QtCore.QModelIndex()
        else:
            return self._treeModel().createIndex(self.treeRow(), 0, self)

    def beginRemoveRow(self, row):
        self._treeModel().beginRemoveRows(self._treeIndex(), row, row)

    def endRemoveRow(self):
        self._treeModel().endRemoveRows()

    def beginInsertRow(self, row):
        self._treeModel().beginInsertRows(self._treeIndex(), row, row)

    def endInsertRow(self):
        self._treeModel().endInsertRows()


class FEMTreeItem(TreeItem):
    def __init__(self, parent=None, fem=None, canvas=None):
        super().__init__(parent)
        self._canvas = canvas
        self._obj = fem

    def fem(self):
        if self._obj is None:
            return self.parent.fem()
        else:
            return self._obj

    def canvas(self):
        if self._canvas is None:
            return self.parent.canvas()
        else:
            return self._canvas
