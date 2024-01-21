from lys_fem import addModel
from lys_fem.ngs import addNGSModel
from .model import MagnetostatisticsModel, Source, DirichletBoundary, InitialCondition
from .ngs import NGSMagnetostatisticsModel

addModel("Electromagnetism", MagnetostatisticsModel)
addNGSModel("Magnetostatistics", NGSMagnetostatisticsModel)
