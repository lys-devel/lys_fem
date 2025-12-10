from .FEM import FEMProject
from .geometry import GeometrySelection
from .mesh import OccMesher
from .material import Material, FEMParameter, materialParameters, UserDefinedParameters
from .model import FEMModel, FEMFixedModel
from .solver import FEMSolver, SolverStep, solvers, StationarySolver, TimeDependentSolver, RelaxationSolver
from .conditions import DomainCondition, BoundaryCondition, InitialCondition
from .solution import FEMSolution, SolutionField
