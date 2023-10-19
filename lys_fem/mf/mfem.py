import datetime
import numpy as np
from lys_fem import mf

if mf.parallel:
    from mfem.par import *
    import mfem.par as mfem_orig
    from mpi4py import MPI

    FiniteElementSpace = mfem_orig.ParFiniteElementSpace
    GridFunction = mfem_orig.ParGridFunction
    BilinearForm = mfem_orig.ParBilinearForm
    LinearForm = mfem_orig.ParLinearForm
    SparseMatrix = mfem_orig.HypreParMatrix

    def print_(*args):
        if mf.parallel:
            if MPI.COMM_WORLD.rank != 0:
                return
        print(*args)

    def getSolver(solver="Default", prec="Default"):
        prec = mfem_orig.HypreSmoother()
        solver = mfem_orig.CGSolver(MPI.COMM_WORLD)
        return solver, prec

    def getData(x, mesh):
        indices = mfem_orig.intArray()
        mesh.GetGlobalVertexIndices(indices)
        indices = np.array([i for i in indices])

        indices = _gatherArray(indices)
        data = _gatherArray(x.GetDataArray())
        if MPI.COMM_WORLD.rank == 0:
            result = np.empty([np.max([np.max(i) for i in indices]) + 1, x.VectorDim()])
            for i, d in zip(indices, data):
                d = d.reshape(x.VectorDim(), -1).T
                result[i] = d
            return result
        else:
            return None

    def _gatherArray(arr):
        sizes = MPI.COMM_WORLD.gather(len(arr), root=0)
        if MPI.COMM_WORLD.rank == 0:
            result = np.empty([len(sizes), max(sizes)], dtype=arr.dtype)
        else:
            result = None
        MPI.COMM_WORLD.Gather(arr, result, root=0)
        if result is None:
            return None
        return [r[:s] for r, s in zip(result, sizes)]

    def saveData(file, data):
        if MPI.COMM_WORLD.rank == 0:
            np.savez(file, **data)

else:
    import mfem.ser as mfem_orig
    from mfem.ser import *
    print_ = print

    def getSolver(solver="Default", prec="Default"):
        prec = mfem_orig.GSSmoother()
        solver = mfem_orig.CGSolver()
        return solver, prec

    def getData(x, mesh):
        return x.GetDataArray().reshape(x.VectorDim(), -1).T

    def saveData(file, data):
        np.savez(file, **data)


def print_initialize():
    print_("\n---------------------Initialization--------------------------")
    if mf.parallel:
        print_("lys_fem starts at", datetime.datetime.now(), " with ", str(MPI.COMM_WORLD.size), "processors")
    else:
        print_("lys_fem starts at", datetime.datetime.now(), "in serial mode")
