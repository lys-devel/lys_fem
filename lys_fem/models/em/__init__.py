from lys_fem import addModel, addMaterialParameter
from lys_fem.ngs import addNGSModel
from ..common import DirichletBoundary, InitialCondition, Source, DivSource, UserDefinedParameter

from .material import ElectrostaticParameters
from .model import MagnetostaticsModel, ElectrostaticsModel
from .ngs import NGSMagnetostaticsModel, NGSElectrostaticsModel

addMaterialParameter("Electromagnetism", ElectrostaticParameters)
addModel("Electromagnetism", MagnetostaticsModel)
addModel("Electromagnetism", ElectrostaticsModel)
addNGSModel("Magnetostatics", NGSMagnetostaticsModel)
addNGSModel("Electrostatics", NGSElectrostaticsModel)
