import ngsolve
from netgen.meshing import Mesh, Element0D, Element1D, Element2D, Element3D, FaceDescriptor, Pnt, MeshPoint

from . import mpi, util


class NGSMesh:
    def __init__(self, mesh):
        self._mesh = mesh

    def update(self, mesh):
        self._mesh = mesh

    def eval(self):
        return self._mesh
    
    @property
    def materials(self):
        return self._mesh.GetMaterials()

    @property
    def boundaries(self):
        return self._mesh.GetBoundaries()


def generateMesh(fem, file="mesh.msh"):
    if mpi.isRoot:
        geom = fem.geometries.generateGeometry()
        fem.mesher.export(geom, file)
        gmesh, tags = ReadGmsh(file, fem.dimension)
        ne, nv = gmesh.ne, len(gmesh.Points())
        gmesh.Scale(fem.geometries.scale)
    else:
        ne, nv = 0, 0

    if mpi.isParallel():
        comm = ngsolve.MPI_Init()
        if mpi.isRoot:
            mesh = ngsolve.Mesh(gmesh.Distribute(comm))
        else:
            mesh = ngsolve.Mesh(Mesh.Receive(comm))
    else:
        mesh = ngsolve.Mesh(gmesh)
    if mpi.isRoot:
        mesh.tags = tags
    mesh = NGSMesh(mesh)
    mesh.ns = ne, nv
    util.dimension = fem.dimension
    util.dx.setMesh(mesh)
    util.ds.setMesh(mesh)
    return mesh


def ReadGmsh(filename, meshdim): #from netgen.read_gmsh import ReadGmsh
    if not filename.endswith(".msh"):
        filename += ".msh"

    f = open(filename, 'r')
    mesh = Mesh(dim=meshdim)

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
                    mesh.Add(Element0D(pointmap[index], index))
                if elem_dim == 1:
                    mesh.Add(Element1D(index=index, vertices=nodenums))
                elif elem_dim==2:
                    mesh.Add(Element2D(index, nodenums))
                elif elem_dim==3:
                    mesh.Add(Element3D(index, nodenums))
                if elem_dim == meshdim:
                    tags.append(tag)

    return mesh, tags
