from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from lys_fem.ngs import addNGSModel
from ..common import InitialCondition

from .material import LLGParameters
from .model import LLGModel, ExternalMagneticField
from .mfem import MFEMLLGModel
from .ngs import NGSLLGModel

addMaterialParameter("Magnetism", LLGParameters)
addModel("Magnetism", LLGModel)
addMFEMModel("LLG", MFEMLLGModel)
addNGSModel("LLG", NGSLLGModel)