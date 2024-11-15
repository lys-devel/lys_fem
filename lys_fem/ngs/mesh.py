import numpy as np
from scipy.spatial import KDTree

import gmsh
import ngsolve
from netgen.meshing import Mesh, Element0D, Element1D, Element2D, Element3D, FaceDescriptor, Pnt, MeshPoint

from . import mpi, util

class NGSMesh(ngsolve.Mesh):
    pass


def loadMesh(fem, file):
    ngmesh = Mesh(dim=fem.dimension)
    ngmesh.Load(file)
    mesh = NGSMesh(ngmesh)
    mesh._coords_global = np.array(ngmesh.Coordinates())
    return mesh


def generateMesh(fem, file="mesh.msh"):
    if mpi.isRoot:
        geom = fem.geometries.generateGeometry()
        fem.mesher.export(geom, file)
        gmesh, points = ReadGmsh(file, fem.dimension)
        gmesh.Scale(fem.geometries.scale)
        if gmesh.dim == 1:
            _createBoundaryFor1D(gmesh, file, points)
        coords = np.array(gmesh.Coordinates())

    if mpi.isParallel():
        from mpi4py import MPI
        if mpi.isRoot:
            mesh = NGSMesh(gmesh.Distribute(MPI.COMM_WORLD))
            mesh._coords_global = coords
        else:
            mesh = NGSMesh(Mesh.Receive(MPI.COMM_WORLD))
        _createMapping(mesh)
    else:
        mesh = NGSMesh(gmesh)
        mesh._coords_global = coords
    util.dimension = fem.dimension
    util.dx.setMesh(mesh)
    util.ds.setMesh(mesh)
    return mesh

def _createBoundaryFor1D(gmesh, file, points):
    # Load file by gmsh
    model = gmsh.model()
    model.add("Default")
    model.setCurrent("Default")
    gmsh.merge(file)

    # Get all boundary nodes
    tags = [tag for dim, tag in model.getEntities(0)]

    # Set the boundary nodes to mesh object.
    for t in tags:
        gmesh.Add(Element0D(points[t], index=t))
        gmesh.SetBCName(t-1, "boundary"+str(t))


def _createMapping(mesh):
    coords_local = mpi.gatherArray(mesh.ngmesh.Coordinates())
    if not mpi.isRoot:
        return
    coords_local = [arr.reshape(-1,mesh.dim) for arr in coords_local] 
    kdtree = KDTree(mesh._coords_global)
    mesh._map_to_global = [kdtree.query(c)[1] for c in coords_local]


def exportMesh(mesh, file=None):
    gmesh = mesh.ngmesh

    def add(d, type, vertices):
        if type not in d:
            d[type] = []
        d[type].append(tuple([v.nr for v in vertices]))

    result = []
    for mat in range(1, 1+ len(mesh.GetMaterials())):
        elements = {}
        if gmesh.dim == 1:
            for e in gmesh.Elements1D():
                if mat == e.index:
                    add(elements, "line", e.vertices)
        if gmesh.dim == 2:
            for e in gmesh.Elements2D():
                if mat == e.index:
                    if len(e.vertices) == 4:
                        add(elements, "quad", e.vertices)
                    if len(e.vertices) == 3:
                        add(elements, "triangle", e.vertices)
        if gmesh.dim == 3:
            for e in gmesh.Elements3D():
                if mat == e.index:
                    if len(e.vertices) == 4:
                        add(elements, "tetra", e.vertices)
                    if len(e.vertices) == 5:
                        add(elements, "pyramid", e.vertices)
                    if len(e.vertices) == 6:
                        add(elements, "prism", e.vertices)
                    if len(e.vertices) == 8:
                        add(elements, "hexa", e.vertices)
        result.append(elements)

    if mpi.isParallel():
        result = _gatherMesh(mesh, result)
    if file is None:
        return result, mesh._coords_global
    if mpi.isRoot:
        np.savez(file, mesh=result, coords=mesh._coords_global)


def _gatherMesh(mesh, result):
    elem_list = {"line": 2, "quad": 4, "triangle": 3, "tetra": 4, "pyramid": 5, "prism": 6, "hexa": 8}
    result_glob = []
    for res_mat in result:
        res_mat_glob = {}
        for key, nnodes in elem_list.items():
            nodes = mpi.gatherArray(res_mat.get(key, []), int, True)
            if mpi.isRoot:
                tmp = np.vstack([map[nodes_local-1].reshape(-1, nnodes) for nodes_local, map in zip(nodes, mesh._map_to_global)])
                if len(tmp) > 0:
                    res_mat_glob[key] = tmp
        result_glob.append(res_mat_glob)
    return result_glob


def ReadGmsh(filename, meshdim): #from netgen.read_gmsh import ReadGmsh
    if not filename.endswith(".msh"):
        filename += ".msh"

    f = open(filename, 'r')
    mesh = Mesh(dim=meshdim)

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
                elmtype, numtags, group, id = int(line[1]), int(line[2]), int(line[3]), int(line[4])
                # the first tag is the physical group nr, the second tag is the group nr of the dim

                if elmtype not in num_nodes_map:
                    raise Exception("element type", elmtype, "not implemented")

                nodenums = line[3 + numtags:3 + numtags + num_nodes_map[elmtype]]
                nodenums = [pointmap[int(nn)] for nn in nodenums]
                nodenums = [nodenums[i] for i in ordering[elmtype]]

                elem_dim = 0 if elmtype==point else (1 if elmtype in [segm, segm3] else (2 if elmtype in [trig, trig6, quad, quad8] else 3))

                if elem_dim == 0:
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

                if elem_dim == 1:
                    mesh.Add(Element1D(index=index, vertices=nodenums))
                elif elem_dim==2:
                    mesh.Add(Element2D(index, nodenums))
                elif elem_dim==3:
                    mesh.Add(Element3D(index, nodenums))

    return mesh, pointmap
