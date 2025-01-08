from lys_fem import addModel, DomainCondition
from lys_fem.ngs import addNGSModel
from ..common import DirichletBoundary, InitialCondition, UserDefinedParameters

from .model import LinearTestModel, NonlinearTestModel, TwoVariableTestModel, ExpTestModel, TdepFieldTestModel, ScaleTestModel, TwoVarGradTestModel, RandomForce, RandomWalkModel
from .ngs import NGSLinearTestModel, NGSNonlinearTestModel, NGSTwoVariableTestModel, NGSExpTestModel, NGSTdepFieldTestModel, NGSScaleTestModel, NGSTwoVarGradTestModel, NGSRandomWalkModel

addModel("Linear Test", LinearTestModel)
addModel("Nonlinear Test", NonlinearTestModel)
addModel("TwoVariable Test", TwoVariableTestModel)
addModel("Exp Test", ExpTestModel)
addModel("TdepField Test", TdepFieldTestModel)
addModel("Scale Test", ScaleTestModel)
addModel("Two Variable Grad Test", TwoVarGradTestModel)
addModel("Random Walk Test", RandomWalkModel)

addNGSModel("Linear Test", NGSLinearTestModel)
addNGSModel("Nonlinear Test", NGSNonlinearTestModel)
addNGSModel("Two Variable Test", NGSTwoVariableTestModel)
addNGSModel("Exp Test", NGSExpTestModel)
addNGSModel("Tdep Field Test", NGSTdepFieldTestModel)
addNGSModel("Scale Test", NGSScaleTestModel)
addNGSModel("Two Variable Grad Test", NGSTwoVarGradTestModel)
addNGSModel("Random Walk Test", NGSRandomWalkModel)
