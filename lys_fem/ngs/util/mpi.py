import numpy as np

def isParallel():
    """
    Return wheather the current process is run in parallel mode.

    Returns:
        bool: Wheather the current process is run in parallel or not.
    """
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


def wait():
    """
    Wait until all process call this function.
    """
    if not isParallel():
        return
    return MPI.COMM_WORLD.gather(0, root=0)


def gatherArray(arr, dtype=float):
    """
    Gather numpy arrays from all process.

    Args:
        arr(numpy array): The numpy arrays to be gathered.

    Returns:
        list of numpy array: The list of arrays gathered.
    """
    if not isParallel():
        return [arr]
    arr = np.array(arr).flatten().astype(dtype)
    sizes = MPI.COMM_WORLD.gather(len(arr), root=0)
    if MPI.COMM_WORLD.rank == 0:
        size = MPI.COMM_WORLD.bcast(max(max(sizes), 1), root=0)
        result = np.empty([len(sizes), max(max(sizes), 1)], dtype=dtype)
    else:
        size = MPI.COMM_WORLD.bcast(None, root=0)
        result = None
    if len(arr) != size:
        arr = np.pad(arr, (0, size-len(arr)))
    MPI.COMM_WORLD.Gather(arr, result, root=0)
    if result is None:
        return None
    return [r[:s] for r, s in zip(result, sizes)]