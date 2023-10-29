from lys_fem import addMaterialParameter, addModel
from lys_fem.mf import addMFEMModel
from .material import ElasticParameters
from .model import ElasticModel
from .mfem import MFEMElasticModel

addMaterialParameter("Acoustics", ElasticParameters)
addModel("Acoustics", ElasticModel)
addMFEMModel("Elasticity", MFEMElasticModel)
