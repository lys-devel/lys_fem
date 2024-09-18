from lys_fem import addMaterialParameter, addModel
from lys_fem.ngs import addNGSModel

from ..common import DirichletBoundary, NeumannBoundary, InitialCondition
from .material import HeatConductionParameters
from .model import HeatConductionModel
from .ngs import NGSHeatConductionModel

addMaterialParameter("Heat", HeatConductionParameters)
addModel("Heat", HeatConductionModel)
addNGSModel("Heat Conduction", NGSHeatConductionModel)
