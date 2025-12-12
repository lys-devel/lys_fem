from lys_fem import addModel, DomainCondition
from ..common import DirichletBoundary, InitialCondition, UserDefinedParameters

from .model import LinearTestModel, NonlinearTestModel, TwoVariableTestModel, ExpTestModel, TdepFieldTestModel, ScaleTestModel, TwoVarGradTestModel, RandomForce, RandomWalkModel

addModel("Test", LinearTestModel)
addModel("Test", NonlinearTestModel)
addModel("Test", TwoVariableTestModel)
addModel("Test", ExpTestModel)
addModel("Test", TdepFieldTestModel)
addModel("Test", ScaleTestModel)
addModel("Test", TwoVarGradTestModel)
addModel("Test", RandomWalkModel)

