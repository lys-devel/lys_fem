import weakref

class FEMObject:
    def __init__(self, parent=None):
        if parent is not None:
            self.setParent(parent)

    @property
    def fem(self):
        return self._fem()

    def setParent(self, parent):
        self._parent = weakref.ref(parent)


class FEMObjectList(list):
    def __init__(self, parent):
        self._parent = weakref.ref(parent)

    def append(self, item):
        super().append(item)
        item.setParent(self._parent())