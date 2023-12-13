import numpy as np
from . import mfem_orig

if mfem_orig.isParallel():
    from mpi4py import MPI

    class MFEMGridFunction(mfem_orig.ParGridFunction):
        def getData(self):
            return _gatherData(self, self.ParFESpace().GetParMesh())

    def _gatherData(x, mesh):
        indices = mfem_orig.intArray()
        mesh.GetGlobalVertexIndices(indices)
        indices = np.array([i for i in indices])
        indices = _gatherArray(indices)

        data_list = []
        v = mfem_orig.Vector()
        for d in range(x.VectorDim()):
            x.GetNodalValues(v, d + 1)
            data_list.append(_gatherArray(v.GetDataArray()))

        if MPI.COMM_WORLD.rank == 0:
            result = np.empty([np.max([np.max(i) for i in indices]) + 1, x.VectorDim()])
            for dim, data in enumerate(data_list):
                for i, d in zip(indices, data):
                    result[i, dim] = d
            return result[:,0]
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

else:
    class MFEMGridFunction(mfem_orig.GridFunction):
        def getData(self):
            res = []
            v = mfem_orig.Vector()
            for d in range(self.VectorDim()):
                self.GetNodalValues(v, d + 1)
                res.append(np.array(v.GetDataArray()))
            return np.array(res).T[:,0]