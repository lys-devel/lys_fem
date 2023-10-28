from lys_fem import addMaterialParameter, addModel
from .material import HeatConductionParameters
from .model import HeatConductionModel

addMaterialParameter("Heat", HeatConductionParameters)
addModel("Heat", HeatConductionModel)
