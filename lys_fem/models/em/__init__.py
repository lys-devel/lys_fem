from lys_fem import addModel, addMaterialParameter
from ..common import DirichletBoundary, InitialCondition, Source, DivSource, UserDefinedParameters

from .material import ElectrostaticParameters
from .model import MagnetostaticsModel, ElectrostaticsModel

addMaterialParameter("Electromagnetism", ElectrostaticParameters)
addModel("Electromagnetism", MagnetostaticsModel)
addModel("Electromagnetism", ElectrostaticsModel)
