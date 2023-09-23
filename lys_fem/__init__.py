from .fem import FEMGeometry, FEMParameter, FEMModel
from .functions import addMaterialParameter, addModel, addGeometry
from . import models


def _makeMenu():
    from lys import glb
    from .gui import FEMGUI
    if glb.mainWindow() is not None:
        menu = glb.mainWindow().menuBar()
        calc = menu.addMenu('Calculators')

        act = calc.addAction("FEM")
        act.triggered.connect(FEMGUI)


_makeMenu()
