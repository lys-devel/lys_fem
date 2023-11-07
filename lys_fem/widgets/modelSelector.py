from lys.Qt import QtWidgets, QtCore


class ModelSelector(QtWidgets.QComboBox):
    def __init__(self, fem):
        super().__init__()
        self._fem = fem
        self.addItems([m.name for m in fem.models])

    def getSelectedModel(self):
        return self._fem.models[self.currentIndex()]

    def setSelectedModel(self, model):
        self.setCurrentIndex(self._fem.models.index(model))
