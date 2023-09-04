from .FEMGUI import FEMGUI


def _makeMenu():
    from lys import glb
    if glb.mainWindow() is not None:
        menu = glb.mainWindow().menuBar()
        calc = menu.addMenu('Calculators')

        act = calc.addAction("FEM")
        act.triggered.connect(FEMGUI)


_makeMenu()
