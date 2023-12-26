from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from lys_fem.ngs import addNGSModel
from .material import HeatConductionParameters
from .model import HeatConductionModel
from .mfem import MFEMHeatConductionModel
from .ngs import NGSHeatConductionModel
from ..common import NeumannBoundary, DirichletBoundary, InitialCondition

addMaterialParameter("Heat", HeatConductionParameters)
addModel("Heat", HeatConductionModel)
addMFEMModel("Heat Conduction", MFEMHeatConductionModel)
addNGSModel("Heat Conduction", NGSHeatConductionModel)
