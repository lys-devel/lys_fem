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

