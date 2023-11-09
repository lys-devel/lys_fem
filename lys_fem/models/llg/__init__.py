from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from .material import LLGParameters
from .model import LLGModel
from .mfem import MFEMLLGModel

addMaterialParameter("Heat", LLGParameters)
addModel("Heat", LLGModel)
addMFEMModel("Heat Conduction", MFEMLLGModel)
