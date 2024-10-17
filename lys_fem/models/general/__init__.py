from lys_fem import addMaterialParameter, addModel
from ..common import Source, DivSource, DirichletBoundary, InitialCondition

from lys_fem.ngs import addNGSModel
from .poisson import PoissonModel, NGSPoissonModel

addModel("General", PoissonModel)
addNGSModel("Poisson", NGSPoissonModel)