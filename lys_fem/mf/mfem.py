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

    def getMesh(mesh, dim):
        mesh = mesh.GetSerialMesh(0)
        if MPI.COMM_WORLD.rank != 0:
            return None
        return _parseMesh(mesh, dim)

    def getData(x, mesh):
        indices = mfem_orig.intArray()
        mesh.GetGlobalVertexIndices(indices)
        indices = np.array([i for i in indices])

        indices = _gatherArray(indices)
        data = _gatherArray(x.GetDataArray())
        if MPI.COMM_WORLD.rank == 0:
            result = np.empty([np.max([np.max(i) for i in indices]) + 1, x.VectorDim()])
            for i, d in zip(indices, data):
                d = d.reshape(-1, x.VectorDim())
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
            np.savez(file, **data, allow_pickle=True)

else:
    import mfem.ser as mfem_orig
    from mfem.ser import *
    print_ = print

    def getSolver(solver="Default", prec="Default"):
        prec = mfem_orig.GSSmoother()
        solver = mfem_orig.CGSolver()
        return solver, prec

    def getMesh(mesh, dim):
        return _parseMesh(mesh, dim)

    def getData(x, mesh):
        return x.GetDataArray().reshape(-1, x.VectorDim())

    def saveData(file, data):
        np.savez(file, **data)


def print_initialize():
    print_("\n---------------------Initialization--------------------------")
    if mf.parallel:
        print_("lys_fem starts at", datetime.datetime.now(), " with ", str(MPI.COMM_WORLD.size), "processors")
    else:
        print_("lys_fem starts at", datetime.datetime.now(), "in serial mode")


def _parseMesh(mesh, dim):
    _key_list = {0: "point", 1: "line", 2: "triangle", 3: "quad", 4: "tetra", 5: "hexa", 6: "prism", 7: "pyramid"}
    _num_list = {"point": 1, "line": 2, "triangle": 3, "quad": 4, "tetra": 4, "hexa": 6, "prism": 6, "pyramid": 5}

    # get coodinates
    coords = mfem_orig.Vector()
    mesh.GetVertices(coords)
    coords = np.array([c for c in coords])
    coords = coords.reshape(dim, -1).T
    coords = np.hstack([coords, np.zeros((len(coords), 3 - dim))])

    # get elements for geometry
    result = {}
    for n, key in _key_list.items():
        # get all vertex list and its geometry for the element type n
        vtx, attr = mfem_orig.intArray(), mfem_orig.intArray()
        mesh.GetElementData(n, vtx, attr)
        vtx = np.array([v for v in vtx]).reshape(-1, _num_list[key])
        attr = np.array([v for v in attr])

        # assign the vertex to result dictionary.
        for i, geom in enumerate(mesh.attributes):
            value = vtx[attr == geom]
            if len(value) == 0:
                continue
            result[key + str(i)] = value

    # reconstruct coords for respective mesh
    for i, geom in enumerate(mesh.attributes):
        nodes_used = set()
        for key in _num_list.keys():
            if key + str(i) in result:
                nodes_used.update(set(result[key + str(i)].flatten()))
        nodes_used = sorted(list(nodes_used))
        for key in _num_list.keys():
            if key + str(i) in result:
                result[key + str(i)] = np.searchsorted(nodes_used, result[key + str(i)])
        result["coords" + str(i)] = coords[nodes_used]
        result["nodes" + str(i)] = np.array(nodes_used)
    return result
