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
else:
    isRoot = True


def gatherArray(arr, dtype=float, test=False):
    if not isParallel():
        return arr
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


file = None

def setOutputFile(path):
    global file
    file = open(path, "w")

    
def print_(*args, **kwargs):
    global file
    if isRoot:
        print(*args, **kwargs)
        if file is not None:
           print(*args, file=file, flush=True, **kwargs)
        

def info():
    if isParallel():
        print_("MPI: ", MPI.COMM_WORLD.size, "processors")
        print_()
    else:
        print_("MPI: No MPI. Run in serial mode.")
        print_()

def wait():
    if not isParallel():
        return
    return MPI.COMM_WORLD.gather(0, root=0)

