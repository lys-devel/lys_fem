from . import mfem_orig

if mfem_orig.isParallel():
    from mpi4py import MPI
    args = [MPI.COMM_WORLD]
else:
    args = []

class FEMSolverBase:
    def __mul__(self, b):
        from .vecmat import MFEMVector
        res = MFEMVector(b.Size())
        self.Mult(b, res)
        return res

class MFEMCGSolver(mfem_orig.CGSolver, FEMSolverBase):
    def __init__(self):
        super().__init__(*args)
        _setSolverParams(self)


class MFEMGMRESSolver(mfem_orig.GMRESSolver, FEMSolverBase):
    def __init__(self):
        super().__init__(*args)
        _setSolverParams(self)


def _setSolverParams(solver, rel_tol=1e-8):
    solver.iterative_mode = False
    solver.SetRelTol(rel_tol)
    solver.SetAbsTol(1e-10)
    solver.SetMaxIter(10000)
    solver.SetPrintLevel(0)
