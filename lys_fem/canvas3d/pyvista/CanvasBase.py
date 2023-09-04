import numpy as np
from lys.Qt import QtWidgets
from pyvistaqt import QtInteractor

from ..interface import CanvasBase3D
from .WaveData import _pyvistaData


class Canvas3d(CanvasBase3D, QtWidgets.QWidget):
    def __init__(self, parent=None):
        CanvasBase3D.__init__(self)
        QtWidgets.QWidget.__init__(self, parent)
        self.__initlayout()
        self.__initCanvasParts()

    def __initlayout(self):
        self._plotter = QtInteractor()
        self._plotter.track_click_position(self.__pick)
        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addWidget(self._plotter.interactor)
        self.setLayout(vlayout)

    def __initCanvasParts(self):
        self.addCanvasPart(_pyvistaData(self))

    @property
    def plotter(self):
        return self._plotter

    def __pick(self, *args):
        print("start pick")
        picked_pt = np.array(self.plotter.pick_mouse_position())
        direction = picked_pt - self.plotter.camera_position[0]
        direction = direction / np.linalg.norm(direction)
        res = self.rayTrace(picked_pt - 1 * direction, picked_pt + 10000 * direction)
        print(res.getWave().note["tag"])
