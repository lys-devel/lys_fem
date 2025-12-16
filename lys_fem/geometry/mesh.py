import gmsh
import numpy as np

from .gmsh import GmshGeometry
from .geometry import ImportGmsh


class GmshMesh:
    _keys = {15: "point", 1: "line", 2: "triangle", 3: "quad", 4: "tetra", 5: "hexa", 6: "prism", 7: "pyramid"}

    def __init__(self, geom, transfinite=None, periodicity=None, size=None, refine=0, generate=True, duplicate=True):
        if not isinstance(geom, GmshGeometry):
            geom = GmshGeometry(geom)
        if duplicate:
            self._geom = geom.duplicate()
        else:
            self._geom = geom
        if generate:
            self._generate(self._geom, transfinite, periodicity, size, refine)

    def _generate(self, geom, transfinite, periodicity, size, refine):
        model = geom.model
        model.setCurrent(geom.name)
        model.mesh.clear()
           
        if transfinite is not None:
            self.__setTransfinite(model, transfinite)
        if periodicity is not None:
            self.__setPeriodicity(model, periodicity)
        if size is not None:
            self.__setSize(model, size, geom.scale)
        model.setCurrent(geom.name)

        model.mesh.generate()
        for _ in range(refine):
            model.mesh.refine()

        model.mesh.optimize()

    def __setTransfinite(self, model, transfinite):
        # prepare partial refinement
        for geom in transfinite:
            dim = {"Volume": 3, "Surface": 2, "Edge": 1, "Point": 0}[geom.geometryType]
            for tag in geom:
                if dim == 2:
                    surf = model.getEntitiesForPhysicalGroup(2, tag)[0]
                    model.mesh.setTransfiniteSurface(surf)
                    model.mesh.setRecombine(2, surf)
                if dim == 3:
                    domain = model.getEntitiesForPhysicalGroup(3, tag)[0]
                    _, surfs = model.getAdjacencies(3, domain)
                    for v in surfs:
                        model.mesh.setTransfiniteSurface(v)
                        model.mesh.setRecombine(2, v)
                    model.mesh.setTransfiniteVolume(domain)

    def __setPeriodicity(self, model, periodicity):
        if len(periodicity) == 0:
            return
        if len(model.getPhysicalGroups(3)) != 0:
            sdim = 2
        else:
            sdim = 1
        ent = [tag for dim, tag in model.getEntities(sdim)]
        for pair in periodicity:
            for p1, p2 in zip(pair[0].getSelection(), pair[1].getSelection()):
                t = self.__getTransform(model, sdim, ent[p1 - 1], ent[p2 - 1])
                model.mesh.setPeriodic(sdim, [ent[p1 - 1]], [ent[p2 - 1]], t)

    def __getTransform(self, model, sdim, e1, e2):
        """
        Consider only parallel shift.
        """
        c1 = np.array([model.getValue(*obj, []) for obj in model.getBoundary([(sdim, e1)], recursive=True)])
        c2 = np.array([model.getValue(*obj, []) for obj in model.getBoundary([(sdim, e2)], recursive=True)])
        for c in c2:
            shift = c1[0] - c
            dist = max([min(np.linalg.norm(c2 - (d - shift), axis=1)) for d in c1])
            if dist < 1e-3:
                return [1, 0, 0, shift[0], 0, 1, 0, shift[1], 0, 0, 1, shift[2], 0, 0, 0, 1]

    def __setSize(self, model, size, scale):
        # size constraint
        for geom in sorted(size, key=lambda x: 1/x.size):
            dim = {"Volume": 3, "Surface": 2, "Edge": 1, "Point": 0}[geom.geometryType]
            for tag in list(geom):
                ents = [(dim, t) for t in model.getEntitiesForPhysicalGroup(dim, tag)]
                if dim == 0:
                    model.mesh.setSize(ents, geom.size/scale)
                else:
                    model.mesh.setSize(model.getBoundary(ents, recursive=True), geom.size/scale)

    @property
    def geometry(self):
        return self._geom

    @property
    def elements(self):
        elem_types, elem_tags, elem_node_tags = self.geometry.mesh.getElements()
        return sum(len(tags) for tags in elem_tags)

    @property
    def nodes(self):
        node_tags, node_coords, _ = self.geometry.mesh.getNodes()
        return len(node_tags)

    @property
    def elementPositions(self):
        mesh = self._geom.model.mesh
        dim = self._geom.dimension
        scale = self._geom.scale

        res = {}
        for etype, etags, enodes in zip(*mesh.getElements(dim=dim)):
            nnodes = mesh.getElementProperties(etype)[3]
            enodes = np.array(enodes).reshape(-1, nnodes)

            for elem_id, nodes in zip(etags, enodes):
                coords = [mesh.getNode(n)[0] for n in nodes]
                coords = np.array(coords).reshape(-1, 3).mean(axis=0)[:dim]
                res[elem_id] = coords/scale
        return res

    def refinedMesh(self, elems, size, amr):
        self.__setGmsh(amr)
        mesh = GmshMesh(self._geom, generate=False)
        alpha, nodes, size = self.__refine(mesh.geometry, self._geom, elems, size, amr, 1.5)

        n = 0
        while nodes < amr.nodes*0.95 or nodes > amr.nodes*1.05:
            alpha, nodes, size = self.__refine(mesh.geometry, self._geom, elems, size, amr, alpha)
            n += 1
            if n > 50:
                break

        gmsh.option.restoreDefaults()
        return mesh

    def __setGmsh(self, amr):
        gmsh.option.setNumber("Mesh.MeshSizeMin", amr.range[0]/self._geom.scale)
        gmsh.option.setNumber("Mesh.MeshSizeMax", amr.range[1]/self._geom.scale)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    def __refine(self, model_new, model_old, elems, size, amr, alpha):
        size *= alpha
        view = gmsh.view.add("refinement")
        gmsh.view.addModelData(view, 0, model_old.name, "ElementData", elems, size)

        mesh = model_new.mesh
        mesh.clear()
        field = mesh.field.add("PostView")
        mesh.field.setNumber(field, "ViewTag", view)
        mesh.field.setAsBackgroundMesh(field)
        mesh.generate()
        mesh.optimize()
        mesh.field.remove(field)
        nodes = len(mesh.getNodes()[0])
        gmsh.view.remove(view)

        return (nodes/amr.nodes)**(1/self._geom.dimension), nodes, size
    
    def getMeshWave(self, dim=3):
        from lys import Wave
        model = self._geom.model
        result = []
        for dim, grp in model.getPhysicalGroups(dim):
            coords_group = np.zeros((0, 3))
            nodes_group = np.zeros((0,), dtype=int)
            elements = {}
            for obj in model.getEntitiesForPhysicalGroup(dim, grp):
                coords, elem, nodes = self.__getMeshForEntity(model, dim, obj)
                coords_group = np.vstack([coords_group, np.reshape(coords, (-1, 3))])
                nodes_group = np.hstack([nodes_group, nodes], dtype=int)
                for type, nodetag in elem.items():
                    key = self._keys[type]
                    if key not in elements:
                        elements[key] = nodetag
                    else:
                        elements[key] = np.vstack([elements[key], nodetag])
            result.append(Wave(np.empty((coords_group.shape[0],)), coords_group, elements=elements, tag=grp, nodes=nodes_group))
        return result

    def __getMeshForEntity(self, model, dim, obj):
        nodes, coords, _ = model.mesh.getNodes(dim, obj, includeBoundary=True)
        sorter = np.argsort(nodes)
        types, _, nodetags = model.mesh.getElements(dim, obj)
        elem = {}
        for type, nodetag in zip(types, nodetags):
            nNodes = model.mesh.getElementProperties(type)[3]
            nodetag = sorter[np.searchsorted(nodes, nodetag, sorter=sorter)]
            nodetag = np.reshape(nodetag, (-1, nNodes))
            elem[type] = nodetag
        return np.reshape(coords, (-1, 3)), elem, nodes
    
    @classmethod
    def fromFile(cls, file, scale):
        geom = GmshGeometry([ImportGmsh(file)], scale=scale)
        return GmshMesh(geom, generate=False, duplicate=False)

    def export(self, file):
        self._geom.export(file)

    def __str__(self):
        res = "Gmsh mesh object, "
        res += str(self.nodes) + " nodes, "
        res += str(self.elements) + " elements"
        return res