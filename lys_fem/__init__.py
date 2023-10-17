import simtools
from .fem import FEMGeometry, FEMParameter, FEMModel, GeometrySelection, FEMSolver
from .functions import addMaterialParameter, addModel, addGeometry, addSolver
from . import models


def _makeMenu():
    from lys import glb
    if glb.mainWindow() is not None:
        from .gui import FEMGUI
        menu = glb.mainWindow().menuBar()
        for m in menu.actions():
            if m.text() == "Calculators":
                act = m.menu().addAction("FEM")
                act.triggered.connect(FEMGUI)


_makeMenu()
