from lys_fem import addModel
from lys_fem.mf import addMFEMModel
from .model import LinearTestModel, NonlinearTestModel
from .mfem import MFEMLinearTestModel, MFEMNonlinearTestModel

#addModel("Test", LinearTestModel)
addMFEMModel("Linear Test", MFEMLinearTestModel)
addMFEMModel("Nonlinear Test", MFEMNonlinearTestModel)
