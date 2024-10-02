from lys_fem import addModel
from lys_fem.ngs import addNGSModel
from ..common import DirichletBoundary, InitialCondition, UserDefinedParameters

from .model import LinearTestModel, NonlinearTestModel, TwoVariableTestModel, ExpTestModel, TdepFieldTestModel
from .ngs import NGSLinearTestModel, NGSNonlinearTestModel, NGSTwoVariableTestModel, NGSExpTestModel, NGSTdepFieldTestModel

addModel("Linear Test", LinearTestModel)
addModel("Nonlinear Test", NonlinearTestModel)
addModel("TwoVariable Test", TwoVariableTestModel)
addModel("Exp Test", ExpTestModel)
addModel("TdepField Test", TdepFieldTestModel)

addNGSModel("Linear Test", NGSLinearTestModel)
addNGSModel("Nonlinear Test", NGSNonlinearTestModel)
addNGSModel("Two Variable Test", NGSTwoVariableTestModel)
addNGSModel("Exp Test", NGSExpTestModel)
addNGSModel("Tdep Field Test", NGSTdepFieldTestModel)
