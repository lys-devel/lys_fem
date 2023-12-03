import datetime
import numpy as np
from scipy.spatial import KDTree


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
    from mfem.par import *
    import mfem.par as mfem_orig
    from mpi4py import MPI

    FiniteElementSpace = mfem_orig.ParFiniteElementSpace
    GridFunction = mfem_orig.ParGridFunction
    BilinearForm = mfem_orig.ParBilinearForm
    MixedBilinearForm = mfem_orig.ParMixedBilinearForm
    LinearForm = mfem_orig.ParLinearForm
    SparseMatrix = mfem_orig.HypreParMatrix
    isRoot = MPI.COMM_WORLD.rank == 0

    def print_(*args):
        if isParallel():
            if MPI.COMM_WORLD.rank != 0:
                return
        print(*args)

    def getSolver(solver="CG", prec=None, rel_tol=1e-8):
        if prec == "GS":
            prec = mfem_orig.HypreBoomerAMG()
        elif prec == "D":
            prec = mfem_orig.HypreSmoother()
        if solver == "CG":
            solver = mfem_orig.CGSolver(MPI.COMM_WORLD)
        elif solver == "GMRES":
            solver = mfem_orig.GMRESSolver(MPI.COMM_WORLD)
        elif solver == "MINRES":
            solver = mfem_orig.MINRESSolver(MPI.COMM_WORLD)
        elif solver == "Newton":
            solver = mfem_orig.NewtonSolver(MPI.COMM_WORLD)
        solver.iterative_mode = False
        solver.SetRelTol(rel_tol)
        solver.SetAbsTol(1e-10)
        solver.SetMaxIter(10000)
        solver.SetPrintLevel(0)
        if prec is not None:
            solver.SetPreconditioner(prec)
        return solver, prec

    def getMesh(mesh):
        mesh = mesh.GetSerialMesh(0)
        if MPI.COMM_WORLD.rank != 0:
            return []
        return _parseMesh(mesh)

    def getData(x, mesh):
        return _gatherData(x, mesh)

    def saveData(file, data):
        if MPI.COMM_WORLD.rank == 0:
            np.savez(file, **data)

    def getMax(data):
        data_array = MPI.COMM_WORLD.gather(data, root=0)
        if isRoot:
            res = MPI.COMM_WORLD.bcast(max(data_array), root=0)
        else:
            res = MPI.COMM_WORLD.bcast(data_array, root=0)
        return res

    def Transpose(mat):
        return mat.Transpose()

    def wait():
        return

    class MFEMBilinearForm(mfem_orig.ParBilinearForm):
        def SpMat(self):
            return self.ParallelAssemble()

    BilinearForm = MFEMBilinearForm
        
    class MFEMMixedBilinearForm(mfem_orig.ParMixedBilinearForm):
        def SpMat(self):
            return self.ParallelAssemble()

    MixedBilinearForm = MFEMMixedBilinearForm
else:
    import mfem.ser as mfem_orig
    from mfem.ser import *
    print_ = print
    isRoot = True

    def getSolver(solver="CG", prec=None, rel_tol=1e-8):
        if prec == "GS":
            prec = mfem_orig.GSSmoother()
        elif prec == "D":
            prec = mfem_orig.DSmoother()
        if solver == "CG":
            solver = mfem_orig.CGSolver()
        elif solver == "GMRES":
            solver = mfem_orig.GMRESSolver()
        elif solver == "MINRES":
            solver = mfem_orig.MINRESSolver()
        elif solver == "Newton":
            solver = mfem_orig.NewtonSolver()
        solver.iterative_mode = False
        solver.SetRelTol(rel_tol)
        solver.SetAbsTol(1e-10)
        solver.SetMaxIter(10000)
        solver.SetPrintLevel(0)
        if prec is not None:
            solver.SetPreconditioner(prec)
        return solver, prec

    def getMesh(mesh):
        return _parseMesh(mesh)

    def getData(x, mesh):
        res = []
        v = mfem_orig.Vector()
        for d in range(x.VectorDim()):
            x.GetNodalValues(v, d + 1)
            res.append(np.array(v.GetDataArray()))
        return np.array(res).T

    def saveData(file, data):
        np.savez(file, **data)

    def getMax(data):
        return data


def print_initialize():
    print_("\n---------------------Initialization--------------------------")
    if isParallel():
        print_("lys_fem starts at", datetime.datetime.now(), " with ", str(MPI.COMM_WORLD.size), "processors")
    else:
        print_("lys_fem starts at", datetime.datetime.now(), "in serial mode")


class _Mesh:
    _key_list = {0: "point", 1: "line", 2: "triangle", 3: "quad", 4: "tetra", 5: "hexa", 6: "prism", 7: "pyramid"}
    _num_list = {"point": 1, "line": 2, "triangle": 3, "quad": 4, "tetra": 4, "hexa": 8, "prism": 6, "pyramid": 5}

    def __init__(self, coordinates, elementGroups, nodes_parent):
        self.elementGroups = elementGroups
        self.coordinates = coordinates
        self.nodes_parent = nodes_parent

    @classmethod
    def fromMFEM(cls, fec, mesh):
        map = mesh.GetParentVertexIDMap()
        if mesh.GetNodes() is None:
            eg = cls.__generateElementGroups(mesh)
            c = cls.__getMeshCoordinates(fec, mesh)
            nodes = np.array([v for v in map])
        else:
            c, eg, nodes = cls.__generateFromNodes(fec, mesh)
            nodes = np.array([map[v] for v in nodes])
        return _Mesh(c, eg, nodes)

    @classmethod
    def __generateElementGroups(cls, mesh):
        elementGroups = {}
        for n, key in cls._key_list.items():
            # get all vertex list and its geometry for the element type n
            vtx = mfem_orig.intArray()
            mesh.GetElementData(n, vtx, mfem_orig.intArray())
            vtx = np.array([v for v in vtx]).reshape(-1, cls._num_list[key])
            if len(vtx) == 0:
                continue
            elementGroups[key] = vtx
        return elementGroups

    @classmethod
    def __getMeshCoordinates(cls, fec, mesh):
        dim = mesh.Dimension()
        space = mfem_orig.FiniteElementSpace(mesh, fec, dim, mfem_orig.Ordering.byVDIM)
        gf = mfem_orig.GridFunction(space)
        mesh.GetNodes(gf)
        result = np.array([c for c in gf.GetDataArray()]).reshape(-1, dim)
        return np.hstack([result, np.zeros((len(result), 3 - dim))])

    @classmethod
    def __generateFromNodes(cls, fec, mesh):
        # get all coordinates in the periodic mesh
        space = mfem_orig.FiniteElementSpace(mesh, fec, mesh.Dimension(), mfem_orig.Ordering.byVDIM)
        coords = mfem_orig.Vector()
        mesh.GetVertices(coords)
        coords = np.array([v for v in coords]).reshape(mesh.Dimension(), -1).T

        # find nodes that is not in the periodic mesh
        kdtree = KDTree(coords)
        coords_new, nodes_new = [], []
        for e in range(mesh.GetNE()):
            vertices = np.array([v for v in mesh.GetElementVertices(e)])
            c = cls.__getVertexPositions(space, e)
            d, _ = kdtree.query(c)
            coords_new.extend(c[d != 0])
            nodes_new.extend(vertices[d != 0])
        coords_new, indices = np.unique(coords_new, return_index=True, axis=0)
        nodes_new = np.array(nodes_new)[indices]
        coords, nodes = np.vstack([coords, coords_new]), np.hstack([np.arange(mesh.GetNV()), nodes_new])

        # Construct element dictionary.
        elements = {}
        kdtree = KDTree(coords)
        for e in range(mesh.GetNE()):
            c = cls.__getVertexPositions(space, e)
            _, id = kdtree.query(c)
            type = cls._key_list[mesh.GetElement(e).GetType()]
            if type not in elements:
                elements[type] = []
            elements[type].append(id)

        return np.hstack([coords, np.zeros((len(coords), 3 - mesh.Dimension()))]), elements, nodes

    @ classmethod
    def __getVertexPositions(self, space, eid):
        mesh = space.GetMesh()
        ir = space.GetFE(eid).GetNodes()
        t = mesh.GetElementTransformation(eid)
        pts = []
        for i in range(ir.GetNPoints()):
            ip = ir.IntPoint(i)
            pts.append(t.Transform(ip))
        return np.array(pts)

    def dictionary(self):
        d = {"coords": self.coordinates, "nodes": self.nodes_parent}
        d.update(self.elementGroups)
        return d


class MFEMVector(mfem_orig.Vector):
    def __add__(self, value):
        res = MFEMVector(value.Size())
        mfem_orig.add_vector(self, 1, value, res)
        return res

    def __sub__(self, value):
        res = MFEMVector(self)
        res -= value
        return res

    def __mul__(self, value):
        res = MFEMVector(self)
        res *= value
        return res

    def __str__(self):
        return str([x for x in self])
    
    def __len__(self):
        return len([x for x in self])


Vector = MFEMVector

class MFEMBlockVector(mfem_orig.BlockVector):
    def __add__(self, value):
        res = MFEMBlockVector(self)
        res += value
        return res

    def __sub__(self, value):
        res = MFEMBlockVector(self)
        res -= value
        return res

    def __mul__(self, value):
        res = MFEMBlockVector(self)
        res *= value
        return res

    def __str__(self):
        return str([x for x in self])
    
    def __len__(self):
        return len([x for x in self])


BlockVector = MFEMBlockVector


class MFEMMatrix(SparseMatrix):
    def __add__(self, value):
        return mfem_orig.Add(1.0, self,1.0, value)

    def __sub__(self, value):
        return mfem_orig.Add(1.0, self,-1.0, value)

    def __neg__(self):
        res = MFEMMatrix(self)
        res *= -1
        return res

    def __mul__(self, value):
        if isinstance(value, MFEMVector):
            res = MFEMVector(value)
            self.Mult(value, res)
            return res
        else:
            res = SparseMatrix(self)
            res *= value
            return res
        
    def __rmul__(self, value):
        return self*value

SparseMatrix = MFEMMatrix

class AddOperator(mfem_orig.PyOperator):
    def __init__(self, op1, op2):
        super().__init__(op1.Height(), op1.Width())
        self._op1 = op1
        self._op2 = op2

    def Mult(self, x, y):
        self._op1.Mult(x,y)
        self._op2.AddMult(x,y)

class MulOperator(mfem_orig.PyOperator):
    def __init__(self, op, value):
        super().__init__(op.Height(), op.Width())
        self._op = op
        self._value = value

    def Mult(self, x, y):
        self._op.Mult(x,y)
        y *= self._value

    def MultTranspose(self, x, y):
        self._op.MultTranspose(x,y)
        y *= self._value

class MFEMBlockOperator(mfem_orig.BlockOperator):
    def __add__(self, value):
        res = MFEMBlockOperator(self.RowOffsets(), self.ColOffsets())
        for r in range(self.NumRowBlocks()):
            for c in range(self.NumColBlocks()):
                if self.IsZeroBlock(r,c):
                    if not value.IsZeroBlock(r,c):
                        res.SetBlock(r,c, value.GetBlock(r,c))
                else:
                    if value.IsZeroBlock(r,c):
                        res.SetBlock(r,c,self.GetBlock(r,c))
                    else:
                        res.SetBlock(r,c,AddOperator(self.GetBlock(r,c), value.GetBlock(r,c)))
        return res

    def __mul__(self, value):
        res = MFEMBlockOperator(self.RowOffsets(), self.ColOffsets())
        for r in range(self.NumRowBlocks()):
            for c in range(self.NumColBlocks()):
                if not self.IsZeroBlock(r,c):
                    res.SetBlock(r,c,MulOperator(self.GetBlock(r,c), value))
        return res

BlockOperator = MFEMBlockOperator


def _parseMesh(mesh):
    fec = mfem_orig.H1_FECollection(1, mesh.Dimension())
    subs = [mfem_orig.SubMesh.CreateFromDomain(mesh, mfem_orig.intArray([index])) for index in mesh.attributes]
    return [_Mesh.fromMFEM(fec, sub) for sub in subs]


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

def wait():
    if isParallel():
        MPI.COMM_WORLD.scatter([0] * MPI.COMM_WORLD.size, root=0)
    else:
        return
