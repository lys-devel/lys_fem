from lys_fem import addMaterialParameter, addModel
from lys_fem.ngs import addNGSModel
from ..common import InitialCondition, DirichletBoundary

from .material import LLGParameters
from .model import LLGModel, LLGEquation, ExternalMagneticField, Demagnetization, UniaxialAnisotropy, GilbertDamping
from .ngs import NGSLLGModel

addMaterialParameter("Magnetism", LLGParameters)
addModel("Magnetism", LLGModel)
addNGSModel("LLG", NGSLLGModel)