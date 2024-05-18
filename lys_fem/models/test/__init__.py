from lys_fem import addModel
from lys_fem.mf import addMFEMModel
from lys_fem.ngs import addNGSModel
from ..common import DirichletBoundary, InitialCondition

from .model import LinearTestModel, NonlinearTestModel, TwoVariableTestModel
from .ngs import NGSLinearTestModel, NGSNonlinearTestModel, NGSTwoVariableTestModel
addModel("Linear Test", LinearTestModel)
addModel("Nonlinear Test", NonlinearTestModel)
addModel("TwoVariable Test", TwoVariableTestModel)

addNGSModel("Linear Test", NGSLinearTestModel)
addNGSModel("Nonlinear Test", NGSNonlinearTestModel)
addNGSModel("Two Variable Test", NGSTwoVariableTestModel)
