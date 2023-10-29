from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from .material import HeatConductionParameters
from .model import HeatConductionModel
from .mfem import MFEMHeatConductionModel

addMaterialParameter("Heat", HeatConductionParameters)
addModel("Heat", HeatConductionModel)
addMFEMModel("Heat Conduction", MFEMHeatConductionModel)
