import numpy as np

def isParallel():
    try:
        from mpi4py import MPI
    except ModuleNotFoundError:
        return False
    if MPI.COMM_WORLD.size == 1:
        return False
    else:
        return True


if isParallel():
    from mpi4py import MPI
    isRoot = MPI.COMM_WORLD.rank == 0
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
else:
    isRoot = True


def allreduce(local):
    if not isParallel():
        return local
    glob = np.zeros_like(local)
    MPI.COMM_WORLD.Allreduce(local, glob, op=MPI.SUM)
    return glob

def bcast(local=None):
    if not isParallel():
        return local
    shape = local.shape if MPI.COMM_WORLD.rank == 0 else None
    MPI.COMM_WORLD.bcast(shape, root=0)
    if MPI.COMM_WORLD.rank != 0:
        local = np.empty(shape)
    return MPI.COMM_WORLD.bcast(local, root=0)