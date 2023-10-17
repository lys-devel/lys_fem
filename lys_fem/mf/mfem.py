import datetime
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

    def Save(x, name):
        x.Save(name + '.{:0>6d}'.format(MPI.COMM_WORLD.rank))

else:
    import mfem.ser as mfem_orig
    from mfem.ser import *
    print_ = print

    def getSolver(solver="Default", prec="Default"):
        prec = mfem_orig.GSSmoother()
        solver = mfem_orig.CGSolver()
        return solver, prec

    def Save(x, name):
        x.Save(name)


def print_initialize():
    print_("\n---------------------Initialization--------------------------")
    if mf.parallel:
        print_("lys_fem starts at", datetime.datetime.now(), " with ", str(MPI.COMM_WORLD.size), "processors")
    else:
        print_("lys_fem starts at", datetime.datetime.now(), "in serial mode")
