from lys_fem import addMaterialParameter, addModel
from .material import ElasticParameters
from .model import ElasticModel

addMaterialParameter("Acoustics", ElasticParameters)
addModel("Acoustics", ElasticModel)
