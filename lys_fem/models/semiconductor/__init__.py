from lys_fem import addMaterialParameter, addModel
from ..common import DirichletBoundary, UserDefinedParameters

from .material import SemiconductorParameters
from .model import SemiconductorModel, InitialCondition

addMaterialParameter("Semiconductor", SemiconductorParameters)
addModel("Semiconductor", SemiconductorModel)
