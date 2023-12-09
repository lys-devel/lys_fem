import datetime
from . import mfem_orig

def print_(*args):
    if mfem_orig.isRoot:
        print(*args)


def print_initialize():
    print_("\n---------------------Initialization--------------------------")
    if mfem_orig.isParallel():
        from mpi4py import MPI
        print_("lys_fem starts at", datetime.datetime.now(), " with ", str(MPI.COMM_WORLD.size), "processors")
    else:
        print_("lys_fem starts at", datetime.datetime.now(), "in serial mode")


def wait():
    if mfem_orig.isParallel():
        from mpi4py import MPI
        MPI.COMM_WORLD.scatter([0] * MPI.COMM_WORLD.size, root=0)
    else:
        return

def getMax(data):
    if mfem_orig.isParallel():
        from mpi4py import MPI
        data_array = MPI.COMM_WORLD.gather(data, root=0)
        if mfem_orig.isRoot:
            res = MPI.COMM_WORLD.bcast(max(data_array), root=0)
        else:
            res = MPI.COMM_WORLD.bcast(data_array, root=0)
        return res
    else:
        return data
