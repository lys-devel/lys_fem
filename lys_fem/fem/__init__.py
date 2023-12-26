from .FEM import FEMProject
from .geometry import FEMGeometry, GeometrySelection
from .mesh import OccMesher
from .material import Material, FEMParameter, materialParameters
from .model import FEMModel, FEMFixedModel
from .solver import FEMSolver, solvers, StationarySolver, TimeDependentSolver
from .equations import Equation
from .domainConditions import DomainCondition, BoundaryCondition
from .initialCondition import InitialCondition
from .solution import FEMSolution
