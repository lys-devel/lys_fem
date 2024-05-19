from .base import FEMCoefficient
from .FEM import FEMProject
from .geometry import FEMGeometry, GeometrySelection
from .mesh import OccMesher
from .material import Material, FEMParameter, materialParameters
from .model import FEMModel, FEMFixedModel
from .solver import FEMSolver, SolverStep, solvers, StationarySolver, TimeDependentSolver, RelaxationSolver
from .equations import Equation
from .conditions import DomainCondition, BoundaryCondition, InitialCondition
from .solution import FEMSolution
