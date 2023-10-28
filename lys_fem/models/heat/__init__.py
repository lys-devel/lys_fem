from lys_fem import addMaterialParameter, addModel
from .material import HeatConductionParameters
#from .model import ElasticModel

addMaterialParameter("Heat", HeatConductionParameters)
#addModel("Acoustics", ElasticModel)
