from lys_fem import addModel, DomainCondition
from ..common import DirichletBoundary, InitialCondition, UserDefinedParameters

from .model import LinearTestModel, NonlinearTestModel, TwoVariableTestModel, ExpTestModel, TdepFieldTestModel, ScaleTestModel, TwoVarGradTestModel, RandomForce, RandomWalkModel

addModel("Linear Test", LinearTestModel)
addModel("Nonlinear Test", NonlinearTestModel)
addModel("TwoVariable Test", TwoVariableTestModel)
addModel("Exp Test", ExpTestModel)
addModel("TdepField Test", TdepFieldTestModel)
addModel("Scale Test", ScaleTestModel)
addModel("Two Variable Grad Test", TwoVarGradTestModel)
addModel("Random Walk Test", RandomWalkModel)

