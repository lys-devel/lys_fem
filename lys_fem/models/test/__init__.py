from lys_fem import addModel
from lys_fem.mf import addMFEMModel
from lys_fem.ngs import addNGSModel
from .model import LinearTestModel, NonlinearTestModel
from .mfem import MFEMLinearTestModel, MFEMNonlinearTestModel
from .ngs import NGSLinearTestModel

#addModel("Test", LinearTestModel)
addMFEMModel("Linear Test", MFEMLinearTestModel)
addMFEMModel("Nonlinear Test", MFEMNonlinearTestModel)

addNGSModel("Linear Test", NGSLinearTestModel)
#addNGSModel("Nonlinear Test", NGSNonlinearTestModel)
