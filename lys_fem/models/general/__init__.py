from lys_fem import addMaterialParameter, addModel
from ..common import Source, DivSource, DirichletBoundary, InitialCondition

from .poisson import PoissonModel

addModel("General", PoissonModel)
