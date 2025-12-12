from lys.Qt import QtWidgets, QtCore, QtGui


class ParameterEditor(QtWidgets.QTreeWidget):
    def __init__(self, param, parent=None):
        super().__init__(parent)
        self._param = param
        self.setColumnCount(2)
        self.setHeaderLabels(["Symbol", "Description"])
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._buildContextMenu)
        self.__initLayout(param)
        self.expandAll()

    def __initLayout(self, param):
        self._widgets = []
        for key, c in param.items():
            if c.valid:
                self.__addItem(key, c)

    def __addItem(self, key, coef):
        child = QtWidgets.QTreeWidgetItem(["", ""])
        parent = QtWidgets.QTreeWidgetItem([key, coef.description])
        parent.addChild(child)
        widget = coef.widget()
        self.addTopLevelItem(parent)
        self.setIndexWidget(self.indexFromItem(child, column=1), widget)
        widget.installEventFilter(self)
        self._widgets.append(widget)

    def eventFilter(self, obj, ev):
        if ev.type() in (QtCore.QEvent.Resize, QtCore.QEvent.LayoutRequest, QtCore.QEvent.Show):
            self.doItemsLayout()
            self.updateGeometries()        
        return False

    def _buildContextMenu(self):
        self._menu = QtWidgets.QMenu()
        sub = self._menu.addMenu("Add")
        for key, coef in self._param.items():
            if not coef.valid:
                sub.addAction(QtWidgets.QAction(key+": "+coef.description, self, triggered=lambda x, y=key,z=coef: self.__add(y,z)))
        self._menu.addAction(QtWidgets.QAction("Remove", self, triggered=self.__remove))
        self._menu.exec_(QtGui.QCursor.pos())

    def __add(self, key, coef):
        coef.setValid()
        self.__addItem(key, coef)

    def __remove(self):
        index = self.indexFromItem(self.currentItem())
        if not index.isValid():
            return
        if index.parent().isValid():
            index=index.parent()
        key = index.data(QtCore.Qt.DisplayRole)
        self._param[key].setValid(False)
        self.takeTopLevelItem(index.row())


class _TreeIndexWidgetSizer(QtCore.QObject):
    def __init__(self, tree, item, col, w):
        super().__init__(w)
        self.tree, self.item, self.col, self.w = tree, item, col, w
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._apply)

        w.installEventFilter(self)
        self._schedule()

    def eventFilter(self, obj, ev):
        if ev.type() in (QtCore.QEvent.Resize, QtCore.QEvent.LayoutRequest, QtCore.QEvent.Show):
            self._schedule()
        return False

    def _schedule(self):
        self._timer.start(0)

    def _apply(self):
        self.item.setSizeHint(self.col, self.w.size())
        self.tree.doItemsLayout()
        self.tree.updateGeometries()