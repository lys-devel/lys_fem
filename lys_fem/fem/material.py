from lys.Qt import QtCore


class MaterialList(QtCore.QObject):
    itemChanged = QtCore.pyqtSignal()

    def __init__(self, materials):
        super().__init__()
        self._list = materials

    def append(self, item):
        self._list.append(item)
        self.itemChanged.emit()

    def remove(self, item):
        self._list.remove(item)
        self.itemChanged.emit()

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __iter__(self):
        return iter(self._list)


class Material(QtCore.QObject):
    itemChanged = QtCore.pyqtSignal()

    def __init__(self, name):
        super().__init__()
        self._name = name
        self._list = []
        self._domains = []

    @property
    def name(self):
        return self._name

    def append(self, item):
        self._list.append(item)
        self.itemChanged.emit()

    def remove(self, item):
        self._list.remove(item)
        self.itemChanged.emit()

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __iter__(self):
        return iter(self._list)

    @property
    def domains(self):
        return self._domains

    @domains.setter
    def domains(self, value):
        self._domains = list(value)


class ElasticParameters:
    def __init__(self, value=0):
        self._func = value
