from lys_fem import addMaterialParameter, addModel
from lys_fem.ngs import addNGSModel

from ..common import DirichletBoundary, InitialCondition, UserDefinedParameters
from .material import ElasticParameters
from .model import ElasticModel, InitialCondition, ThermoelasticStress, DeformationPotential
from .ngs import NGSElasticModel

addMaterialParameter("Acoustics", ElasticParameters)
addModel("Acoustics", ElasticModel)
addNGSModel("Elasticity", NGSElasticModel)
