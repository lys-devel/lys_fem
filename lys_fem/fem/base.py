import weakref

class FEMObject:
    @property
    def fem(self):
        from .FEM import FEMProject
        obj = self
        while not isinstance(obj, FEMProject):
            obj = obj.parent
        return obj

    @property
    def parent(self):
        return self._parent()

    def setParent(self, parent):
        self._parent = weakref.ref(parent)


class FEMObjectList(list, FEMObject):
    def __init__(self, parent, items=[]):
        super().__init__(items)
        self.setParent(parent)
        for item in items:
            if isinstance(item, FEMObject):
                item.setParent(self)

    def append(self, item):
        super().append(item)
        item.setParent(self)