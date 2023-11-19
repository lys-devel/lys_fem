from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from .material import LLGParameters
from .model import LLGModel
from .mfem2 import MFEMLLGModel

addMaterialParameter("Magnetism", LLGParameters)
addModel("Magnetism", LLGModel)
addMFEMModel("LLG", MFEMLLGModel)
