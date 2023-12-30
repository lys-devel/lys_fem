from lys_fem.fem import InitialCondition
from lys_fem import addMaterialParameter

from .meshParams import InfiniteVolumeParams
from .domainConditions import Source
from .boundaryConditions import NeumannBoundary, DirichletBoundary


addMaterialParameter("Deformation", InfiniteVolumeParams)