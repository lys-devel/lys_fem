from lys_fem import addMaterialParameter, addModel
from ..common import InitialCondition, DirichletBoundary, UserDefinedParameters

from .material import LLGParameters
from .model import LLGModel, ExternalMagneticField, UniaxialAnisotropy, CubicAnisotropy, MagneticScalarPotential, SpinTransferTorque, ThermalFluctuation, CubicMagnetoStriction, CubicMagnetoRotationCoupling, BarnettEffect

addMaterialParameter("Magnetism", LLGParameters)
addModel("Magnetism", LLGModel)
