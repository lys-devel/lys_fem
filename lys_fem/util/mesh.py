import numpy as np
import ngsolve

from netgen.meshing import Element0D, Element1D, Element2D, Element3D, FaceDescriptor, Pnt, MeshPoint
from netgen.meshing import Mesh as netgenMesh
from lys_fem.geometry import GmshMesh
from . import mpi

class Mesh(ngsolve.Mesh):
    def __init__(self, model, scale=None):
        if isinstance(model, str):
            model = GmshMesh.fromFile(model, scale)
        self._model = model
        if mpi.isRoot:
            scale = model.geometry.scale
            dimension = model.geometry.dimension

            model.export("tmp.msh")
            gmesh, _ = ReadGmshFile("tmp.msh", dimension) # from python model

            gmesh.Scale(scale)

        if mpi.isParallel():
            comm = ngsolve.MPI_Init()
            if mpi.isRoot:
                pmesh = gmesh.Distribute(comm)
            else:
                pmesh = netgenMesh.Receive(comm)
        else:
            pmesh = gmesh

        super().__init__(pmesh)

    @property
    def elements(self):
        if mpi.isRoot:
            return self._model.elements
        else:
            return 0

    @property
    def nodes(self):
        if mpi.isRoot:
            return self._model.nodes
        else:
            return 0
            
    def refinedMesh(self, error, amr):
        def get_error(x=None):
            x = mpi.bcast(np.array(x))
            res = np.nan_to_num(error(x), 0)
            return mpi.allreduce(res)

        if mpi.isRoot:
            pos = self._model.elementPositions
            err = get_error(list(pos.values()))
            size = 0.2*(err/np.median(err) + 1e-6)**(-0.4)
            mesh_new = self._model.refinedMesh(list(pos.keys()), np.array([size]).T, amr)
        else:
            err = get_error()
            mesh_new = None
        return Mesh(mesh_new, self._model.geometry.scale)

    def save(self, path):
        if mpi.isRoot:
            self._model.export(path)

    def __str__(self):
        res = "Mesh object: "
        res += str(self.nodes) + " nodes, "
        res += str(self.elements) + " elements."
        return res


def ReadGmshModel(geom, meshdim): #from netgen.read_gmsh import ReadGmsh
    mesh = netgenMesh(dim=meshdim)
    __loadGroups(mesh, geom, meshdim)
    _tag = __loadMesh(mesh, geom, meshdim)
    return mesh, _tag


def __loadGroups(mesh, geom, meshdim):
    for obj in geom.getPhysicalGroups(meshdim):
        mesh.SetMaterial(obj[1], "domain"+str(obj[1]))

    for obj in geom.getPhysicalGroups(meshdim-1):
        fd = FaceDescriptor(bc=obj[1])
        fd.bcname = 'boundary' + str(obj[1])
        mesh.SetBCName(obj[1]-1, 'boundary' + str(obj[1]))
        mesh.Add(fd)

def __loadMesh(mesh, geom, meshdim):
    nodeMap = {}
    registered = {}
    node_tags, coords, _ = geom.mesh.getNodes(dim=meshdim, includeBoundary=True)
    for n, c in zip(node_tags, coords.reshape(-1,3)):
        c = tuple(c)
        if c in registered:
            nodeMap[n] = registered[c]
        else:
            index = mesh.Add(MeshPoint(Pnt(*c)))
            nodeMap[n] = index
            registered[c] = index

    for g in geom.getPhysicalGroups(meshdim-1):
        for ent in geom.getEntitiesForPhysicalGroup(*g):
            for type, tags, nodes in zip(*geom.mesh.getElements(dim=meshdim-1, tag=ent)):
                for el in __parseElements(type, np.array([nodeMap[n] for n in nodes]), g[1]):
                    mesh.Add(el)

    _tag = []
    for g in geom.getPhysicalGroups(meshdim):
        for ent in geom.getEntitiesForPhysicalGroup(*g):
            for type, tags, nodes in zip(*geom.mesh.getElements(dim=meshdim, tag=ent)):
                _tag.extend(tags)
                for el in __parseElements(type, np.array([nodeMap[n] for n in nodes]), g[1]):
                    mesh.Add(el)


def __parseElements(type, nodes, index):
    if type == 15: # point
        return [Element0D(index, index) for n in nodes]
    elif type == 1: # line
        return [Element1D(index=index, vertices=n.tolist()) for n in nodes.reshape(-1, 2)]
    elif type == 2: # triangle
        return [Element2D(index, n) for n in nodes.reshape(-1, 3)]
    elif type == 3: # quad
        return [Element2D(index, n) for n in nodes.reshape(-1, 4)]
    elif type == 4: # tetra
        return [Element3D(index, n) for n in nodes.reshape(-1, 4)]
    elif type == 5: # hex
        return [Element3D(index, n[[0,1,5,4,3,2,6,7]]) for n in nodes.reshape(-1, 8)]
    else:
        raise RuntimeError("error!", type, nodes, index)


def ReadGmshFile(filename, meshdim): #from netgen.read_gmsh import ReadGmsh
    if not filename.endswith(".msh"):
        filename += ".msh"

    f = open(filename, 'r')
    mesh = netgenMesh(dim=meshdim)

    tags = []
    pointmap, facedescriptormap, materialmap, bbcmap = {}, {}, {}, {}

    point, segm, trig, quad, tet, hex, prism, pyramid = 15, 1, 2, 3, 4, 5, 6, 7
    segm3, trig6, tet10, quad8, hex20, prism15, pyramid13 = 8, 9, 11, 16, 17, 18, 19      # 2nd order line, trig, tet, quad, hex, prism, pyramid

    num_nodes_map = { segm : 2, trig : 3, quad : 4, tet : 4, hex : 8, prism : 6, pyramid : 5, segm3 : 3, trig6 : 6, tet10 : 10, point : 1, quad8 : 8, hex20 : 20, prism15 : 18, pyramid13 : 19 }
    ordering = {point: [0], segm: [0,1], segm3: [0,1,2], trig: [0,1,2], trig6: [0,1,2,4,5,3], quad: [0,1,2,3], quad8: [0,1,2,3,4,6,7,5],
                tet: [0,1,2,3], tet10: [0,1,2,3,4,6,7,5,9,8], hex: [0,1,5,4,3,2,6,7], hex20: [0,1,5,4,3,2,6,7,8,16,10,12,13,19,15,14,9,11,18,17],
                prism: [0,2,1,3,5,4], prism15: [0,2,1,3,5,4,7,6,9,8,11,10,13,12,14], pyramid: [3,2,1,0,4], pyramid13: [3,2,1,0,4,10,5,6,8,12,11,9,7]}

    while True:
        line = f.readline()
        if line == "":
            break

        if line.split()[0] == "$PhysicalNames":
            for _ in range(int(f.readline())):
                line = f.readline()

        if line.split()[0] == "$Nodes":
            for _ in range(int(f.readline().split()[0])):
                line = f.readline()
                nodenum, x, y, z = line.split()[0:4]
                pnum = mesh.Add(MeshPoint(Pnt(float(x), float(y), float(z))))
                pointmap[int(nodenum)] = pnum

        if line.split()[0] == "$Elements":
            for _ in range(int(f.readline().split()[0])):
                line = f.readline().split()
                tag, elmtype, numtags, group, id = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])
                # the first tag is the physical group nr, the second tag is the group nr of the dim

                if elmtype not in num_nodes_map:
                    raise Exception("element type", elmtype, "not implemented")

                nodenums = line[3 + numtags:3 + numtags + num_nodes_map[elmtype]]
                nodenums = [pointmap[int(nn)] for nn in nodenums]
                nodenums = [nodenums[i] for i in ordering[elmtype]]

                elem_dim = 0 if elmtype==point else (1 if elmtype in [segm, segm3] else (2 if elmtype in [trig, trig6, quad, quad8] else 3))

                if elem_dim == 0 and meshdim != 1:
                    continue

                if elem_dim == meshdim:
                    if id in materialmap:
                        index = materialmap[id]
                    else:
                        index = len(materialmap) + 1
                        mesh.SetMaterial(index, "domain" + str(group))
                        materialmap[id] = index

                if elem_dim == meshdim -1:
                    if id in facedescriptormap.keys():
                        index = facedescriptormap[id]
                    else:
                        index = len(facedescriptormap) + 1
                        fd = FaceDescriptor(bc=index)
                        fd.bcname = 'boundary' + str(group)
                        mesh.SetBCName(index - 1, 'boundary' + str(group))
                        mesh.Add(fd)
                        facedescriptormap[id] = index

                if elem_dim == meshdim -2:
                    if id in bbcmap:
                        index = bbcmap[id]
                    else:
                        index = len(bbcmap) + 1
                        mesh.SetCD2Name(index, "edge" + str(group))
                        bbcmap[id] = index

                if elem_dim == 0:
                    el = Element0D(pointmap[index], index)
                if elem_dim == 1:
                    el = Element1D(index=index, vertices=nodenums)
                elif elem_dim==2:
                    el = Element2D(index, nodenums)
                elif elem_dim==3:
                    el = Element3D(index, nodenums)
                mesh.Add(el)
                if elem_dim == meshdim:
                    tags.append(tag)

    return mesh, tags
