from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from lys_fem.ngs import addNGSModel

from ..common import DirichletBoundary, InitialCondition
from .material import ElasticParameters
from .model import ElasticModel
from .mfem import MFEMElasticModel
from .ngs import NGSElasticModel

addMaterialParameter("Acoustics", ElasticParameters)
addModel("Acoustics", ElasticModel)
addMFEMModel("Elasticity", MFEMElasticModel)
addNGSModel("Elasticity", NGSElasticModel)
