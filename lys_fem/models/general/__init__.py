from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from .poisson import PoissonModel, MFEMPoissonModel

addModel("General", PoissonModel)
addMFEMModel("Poisson", MFEMPoissonModel)
