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
