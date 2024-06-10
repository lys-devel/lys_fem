from lys_fem import addModel, addMaterialParameter
from lys_fem.ngs import addNGSModel
from ..common import DirichletBoundary

from .material import ElectrostaticParameters
from .model import MagnetostaticsModel, MagnetostaticInitialCondition, MagnetostaticSource
from .model import ElectrostaticsModel, ElectrostaticInitialCondition, ElectrostaticSource
from .ngs import NGSMagnetostaticsModel, NGSElectrostaticsModel

addMaterialParameter("Electromagnetism", ElectrostaticParameters)
addModel("Electromagnetism", MagnetostaticsModel)
addModel("Electromagnetism", ElectrostaticsModel)
addNGSModel("Magnetostatics", NGSMagnetostaticsModel)
addNGSModel("Electrostatics", NGSElectrostaticsModel)
