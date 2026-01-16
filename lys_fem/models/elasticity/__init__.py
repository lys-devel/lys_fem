from lys_fem import addMaterialParameter, addModel

from ..common import DirichletBoundary, NeumannBoundary, InitialCondition, UserDefinedParameters
from .material import ElasticParameters
from .model import ElasticModel, InitialCondition, ThermoelasticStress, DeformationPotential, PerfectlyMatchedLayer, InversePiezoelectricity

addMaterialParameter("Acoustics", ElasticParameters)
addModel("Acoustics", ElasticModel)
