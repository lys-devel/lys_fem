from lys_fem import addMaterialParameter, addModel
from lys_fem.ngs import addNGSModel
from ..common import DirichletBoundary, UserDefinedParameters

from .material import SemiconductorParameters
from .model import SemiconductorModel, InitialCondition
from .ngs import NGSSemiconductorModel

addMaterialParameter("Semiconductor", SemiconductorParameters)
addModel("Semiconductor", SemiconductorModel)
addNGSModel("Semiconductor", NGSSemiconductorModel)