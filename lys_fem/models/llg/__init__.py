from lys_fem import addMaterialParameter, addModel
from lys_fem.ngs import addNGSModel
from ..common import InitialCondition, DirichletBoundary, UserDefinedParameters

from .material import LLGParameters
from .model import LLGModel, LLGEquation, ExternalMagneticField, UniaxialAnisotropy, CubicAnisotropy, MagneticScalarPotential, SpinTransferTorque, ThermalFluctuation, CubicMagnetoStriction, CubicMagnetoRotationCoupling, BarnettEffect
from .ngs import NGSLLGModel

addMaterialParameter("Magnetism", LLGParameters)
addModel("Magnetism", LLGModel)
addNGSModel("LLG", NGSLLGModel)