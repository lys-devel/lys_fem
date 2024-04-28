from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from lys_fem.ngs import addNGSModel
from ..common import DirichletBoundary
from .material import HeatConductionParameters
from .model import HeatConductionModel, NeumannBoundary, InitialCondition
from .mfem import MFEMHeatConductionModel
from .ngs import NGSHeatConductionModel

addMaterialParameter("Heat", HeatConductionParameters)
addModel("Heat", HeatConductionModel)
addMFEMModel("Heat Conduction", MFEMHeatConductionModel)
addNGSModel("Heat Conduction", NGSHeatConductionModel)
