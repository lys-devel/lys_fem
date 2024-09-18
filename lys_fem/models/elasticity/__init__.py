from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from lys_fem.ngs import addNGSModel

from ..common import DirichletBoundary, InitialCondition
from .material import ElasticParameters, ThermalExpansionParameters, DeformationPotentialParameters
from .model import ElasticModel, ChristffelEquation, InitialCondition, ThermoelasticStress, DeformationPotential
from .mfem import MFEMElasticModel
from .ngs import NGSElasticModel

addMaterialParameter("Acoustics", ElasticParameters)
addMaterialParameter("Acoustics", ThermalExpansionParameters)
addMaterialParameter("Acoustics", DeformationPotentialParameters)
addModel("Acoustics", ElasticModel)
addMFEMModel("Elasticity", MFEMElasticModel)
addNGSModel("Elasticity", NGSElasticModel)
