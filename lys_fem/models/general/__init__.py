from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from lys_fem.ngs import addNGSModel
from .poisson import PoissonModel, MFEMPoissonModel, NGSPoissonModel

addModel("General", PoissonModel)
addMFEMModel("Poisson", MFEMPoissonModel)
addNGSModel("Poisson", NGSPoissonModel)