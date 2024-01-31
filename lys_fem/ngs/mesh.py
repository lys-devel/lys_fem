import numpy as np
from scipy.spatial import KDTree

import gmsh
import ngsolve
from netgen.meshing import Mesh, Element0D, Element1D, Element2D, Element3D, FaceDescriptor, Pnt, MeshPoint

from . import mpi

class NGSMesh(ngsolve.Mesh):
    pass

def generateMesh(fem, file="mesh.msh"):
    if mpi.isRoot:
        geom = fem.geometries.generateGeometry()
        fem.mesher.export(geom, file)
        gmesh, points = ReadGmsh(file)
        if gmesh.dim == 1:
            _createBoundaryFor1D(gmesh, file, points)
        _setBCNames(gmesh, file)
        coords = np.array(gmesh.Coordinates())
        help(gmesh)

    if mpi.isParallel():
        from mpi4py import MPI
        if mpi.isRoot:
            mesh = NGSMesh(gmesh.Distribute(MPI.COMM_WORLD))
            mesh._coords_global = coords * fem.scaling.getScaling("m")
        else:
            mesh = NGSMesh(Mesh.Receive(MPI.COMM_WORLD))
        _createMapping(mesh)
    else:
        mesh = NGSMesh(gmesh)
        mesh._coords_global = coords * fem.scaling.getScaling("m")
    return mesh

def _setBCNames(gmesh, file):
    # Load file by gmsh
    model = gmsh.model()
    model.add("Default")
    model.setCurrent("Default")
    gmsh.merge(file)
    for dim, tag in model.getEntities(gmesh.dim-1):
        gmesh.SetBCName(tag-1, "boundary"+str(tag))

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


def exportMesh(mesh, file):
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
    if mpi.isRoot:
        np.savez(file, mesh=result, coords=mesh._coords_global)


def exportSolution(mesh, solution, file):
    if not mpi.isParallel():
        return np.savez(file, **solution)
    result = {}
    for key, sol_loc in solution.items():
        sol_loc = mpi.gatherArray(sol_loc)
        if mpi.isRoot:
            sol_glob = np.empty((len(mesh._coords_global)))
            for map, sol in zip(mesh._map_to_global, sol_loc):
                sol_glob[map] = sol
            result[key] = sol_glob
    if mpi.isRoot:
        np.savez(file, **result)


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


def ReadGmsh(filename): #from netgen.read_gmsh import ReadGmsh
    if not filename.endswith(".msh"):
        filename += ".msh"
    meshdim = 1
    with open(filename, 'r') as f:
        while f.readline().split()[0] != "$Elements":
            pass
        nelem = int(f.readline())
        for i in range(nelem):
            line = f.readline()
            eltype = int(line.split()[1])
            if eltype > 1 and eltype != 15:
                meshdim = 2
            if eltype > 3 and eltype != 15:
                meshdim = 3
                break

    f = open(filename, 'r')
    mesh = Mesh(dim=meshdim)

    pointmap = {}
    facedescriptormap = {}
    namemap = { 0 : "default" }
    materialmap = {}
    bbcmap = {}

    segm = 1
    trig = 2
    quad = 3
    tet = 4
    hex = 5
    prism = 6
    pyramid = 7
    segm3 = 8      # 2nd order line
    trig6 = 9      # 2nd order trig
    tet10 = 11     # 2nd order tet
    point = 15
    quad8 = 16     # 2nd order quad
    hex20 = 17     # 2nd order hex
    prism15 = 18   # 2nd order prism
    pyramid13 = 19 # 2nd order pyramid
    segms = [segm, segm3]
    trigs = [trig, trig6]
    quads = [quad, quad8]
    tets = [tet, tet10]
    hexes = [hex, hex20]
    prisms = [prism, prism15]
    pyramids = [pyramid, pyramid13]
    elem0d = [point]
    elem1d = segms
    elem2d = trigs + quads
    elem3d = tets + hexes + prisms + pyramids

    num_nodes_map = { segm : 2,
                      trig : 3,
                      quad : 4,
                      tet : 4,
                      hex : 8,
                      prism : 6,
                      pyramid : 5,
                      segm3 : 3,
                      trig6 : 6,
                      tet10 : 10,
                      point : 1,
                      quad8 : 8,
                      hex20 : 20,
                      prism15 : 18,
                      pyramid13 : 19 }

    while True:
        line = f.readline()
        if line == "":
            break

        if line.split()[0] == "$PhysicalNames":
            #print('WARNING: Physical groups detected - Be sure to define them for every geometrical entity.')
            numnames = int(f.readline())
            for i in range(numnames):
                f.readline
                line = f.readline()
                namemap[int(line.split()[1])] = line.split()[2][1:-1]

        if line.split()[0] == "$Nodes":
            num = int(f.readline().split()[0])
            for i in range(num):
                line = f.readline()
                nodenum, x, y, z = line.split()[0:4]
                pnum = mesh.Add(MeshPoint(Pnt(float(x), float(y), float(z))))
                pointmap[int(nodenum)] = pnum

        if line.split()[0] == "$Elements":
            num = int(f.readline().split()[0])

            for i in range(num):
                line = f.readline().split()
                elmnum = int(line[0])
                elmtype = int(line[1])
                numtags = int(line[2])
                # the first tag is the physical group nr, the second tag is the group nr of the dim
                tags = [int(line[3 + k]) for k in range(numtags)]

                if elmtype not in num_nodes_map:
                    raise Exception("element type", elmtype, "not implemented")
                num_nodes = num_nodes_map[elmtype]

                nodenums = line[3 + numtags:3 + numtags + num_nodes]
                nodenums2 = [pointmap[int(nn)] for nn in nodenums]

                if elmtype in elem1d:
                    if meshdim == 3:
                        if tags[1] in bbcmap:
                            index = bbcmap[tags[1]]
                        else:
                            index = len(bbcmap) + 1
                            if len(namemap):
                                mesh.SetCD2Name(index, namemap[tags[0]])
                            else:
                                mesh.SetCD2Name(index, "line" + str(tags[1]))
                            bbcmap[tags[1]] = index

                    elif meshdim == 2:
                        if tags[1] in facedescriptormap.keys():
                            index = facedescriptormap[tags[1]]
                        else:
                            index = len(facedescriptormap) + 1
                            fd = FaceDescriptor(bc=index)
                            if len(namemap):
                                fd.bcname = namemap[tags[0]]
                            else:
                                fd.bcname = 'line' + str(tags[1])
                            mesh.SetBCName(index - 1, fd.bcname)
                            mesh.Add(fd)
                            facedescriptormap[tags[1]] = index
                    else:
                        if tags[1] in materialmap:
                            index = materialmap[tags[1]]
                        else:
                            index = len(materialmap) + 1
                            if len(namemap):
                                mesh.SetMaterial(index, namemap[tags[0]])
                            else:
                                mesh.SetMaterial(index, "line" + str(tags[1]))
                            materialmap[tags[1]] = index

                    mesh.Add(Element1D(index=index, vertices=nodenums2))

                if elmtype in elem2d:  # 2d elements
                    if meshdim == 3:
                        if tags[1] in facedescriptormap.keys():
                            index = facedescriptormap[tags[1]]
                        else:
                            index = len(facedescriptormap) + 1
                            fd = FaceDescriptor(bc=index)
                            if len(namemap):
                                fd.bcname = namemap[tags[0]]
                            else:
                                fd.bcname = "surf" + str(tags[1])
                            mesh.SetBCName(index - 1, fd.bcname)
                            mesh.Add(fd)
                            facedescriptormap[tags[1]] = index
                    else:
                        if tags[1] in materialmap:
                            index = materialmap[tags[1]]
                        else:
                            index = len(materialmap) + 1
                            if len(namemap):
                                mesh.SetMaterial(index, namemap[tags[0]])
                            else:
                                mesh.SetMaterial(index, "surf" + str(tags[1]))
                            materialmap[tags[1]] = index

                    if elmtype in trigs:
                        ordering = [i for i in range(3)]
                        if elmtype == trig6:
                            ordering += [4,5,3]
                    if elmtype in quads:
                        ordering = [i for i in range(4)]
                        if elmtype == quad8:
                            ordering += [4, 6, 7, 5]
                    mesh.Add(Element2D(index, [nodenums2[i] for i in ordering]))

                if elmtype in elem3d:  # volume elements
                    if tags[1] in materialmap:
                        index = materialmap[tags[1]]
                    else:
                        index = len(materialmap) + 1
                        if len(namemap):
                            mesh.SetMaterial(index, namemap[tags[0]])
                        else:
                            mesh.SetMaterial(index, "vol" + str(tags[1]))
                        materialmap[tags[1]] = index

                    nodenums2 = [pointmap[int(nn)] for nn in nodenums]

                    if elmtype in tets:
                        ordering = [0,1,2,3]
                        if elmtype == tet10:
                            ordering += [4,6,7,5,9,8]
                    elif elmtype in hexes:
                        ordering = [0,1,5,4,3,2,6,7]
                        if elmtype == hex20:
                            ordering += [8,16,10,12,13,19,15,14,9,11,18,17]
                    elif elmtype in prisms:
                        ordering = [0,2,1,3,5,4]
                        if elmtype == prism15:
                            ordering += [7,6,9,8,11,10,13,12,14]
                    elif elmtype in pyramids:
                        ordering = [3,2,1,0,4]
                        if elmtype == pyramid13:
                            ordering += [10,5,6,8,12,11,9,7]
                    mesh.Add(Element3D(index, [nodenums2[i] for i in ordering]))

    return mesh, pointmap
