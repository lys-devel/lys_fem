from .mfem_orig import isParallel, isRoot
from .vecmat import MFEMVector, MFEMBlockVector, MFEMMatrix, MFEMBlockOperator
from .forms import MFEMBilinearForm, MFEMMixedBilinearForm, MFEMLinearForm
from .solvers import MFEMCGSolver,  MFEMGMRESSolver
from .gridFunction import MFEMGridFunction
from .functions import print_, print_initialize, wait, getMax
from .mesh import getMesh

if isParallel():
    import mfem.par as mfem_orig
    from mfem.par import *
    FiniteElementSpace = mfem_orig.ParFiniteElementSpace

else:
    import mfem.ser as mfem_orig
    from mfem.ser import *


Vector = MFEMVector
BlockVector = MFEMBlockVector
SparseMatrix = MFEMMatrix
BlockOperator = MFEMBlockOperator

GridFunction = MFEMGridFunction

LinearForm = MFEMLinearForm
BilinearForm = MFEMBilinearForm
MixedBilinearForm = MFEMMixedBilinearForm

CGSolver = MFEMCGSolver
GMRESSolver = MFEMGMRESSolver

from .coef import generateCoefficient, GridFunctionCoefficient, ConstantCoefficient, SumCoefficient, ProductCoefficient, PowerCoefficient, VectorArrayCoefficient, MatrixArrayCoefficient, SympyCoefficient



