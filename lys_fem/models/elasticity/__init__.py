from lys_fem import addMaterialParameter, addModel

from ..common import DirichletBoundary, InitialCondition, UserDefinedParameters
from .material import ElasticParameters
from .model import ElasticModel, InitialCondition, ThermoelasticStress, DeformationPotential, PerfectlyMatchedLayer

addMaterialParameter("Acoustics", ElasticParameters)
addModel("Acoustics", ElasticModel)
