from . import mfem_orig
from scipy.spatial import KDTree
import numpy as np

if mfem_orig.isParallel():
    from mpi4py import MPI
    def getMesh(mesh):
        smesh = mesh.GetSerialMesh(0)
        if MPI.COMM_WORLD.rank != 0:
            return []
        return _parseMesh(smesh)

else:
    def getMesh(mesh):
        return _parseMesh(mesh)


def _parseMesh(mesh):
    fec = mfem_orig.H1_FECollection(1, mesh.Dimension())
    subs = [mfem_orig.SubMesh.CreateFromDomain(mesh, mfem_orig.intArray([index])) for index in mesh.attributes]
    return [_Mesh.fromMFEM(fec, sub) for sub in subs]


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
