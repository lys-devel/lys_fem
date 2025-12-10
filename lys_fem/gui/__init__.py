from .FEMGUI import FEMGUI
from .conditionGUI import ConditionWidget
from .modelGUI import FEMModelWidget, FEMFixedModelWidget, MethodComboBox
from .solverGUI import StationarySolverWidget, TimeDependentSolverWidget, RelaxationSolverWidget

def _makeMenu():
    from lys import glb
    if glb.mainWindow() is not None:
        menu = glb.mainWindow().menuBar()
        for m in menu.actions():
            if m.text() == "Calculators":
                act = m.menu().addAction("FEM")
                act.triggered.connect(FEMGUI)

_makeMenu()
