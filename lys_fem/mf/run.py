
from . import mfem
from .mesh import generateMesh
from .material import generateMaterial
from .models import generateModel
from .solver import generateSolver


def run(fem):
    mfem.print_initialize()
    geom = fem.geometryGenerator.generateGeometry(fem.dimension)
    fem.mesher.export(geom, "mesh.msh")
    mesh, nv = generateMesh("mesh.msh")
    mfem.print_("Mesh generated: ", str(len([1 for _ in mesh.attributes])), "domains,", str(len([1 for _ in mesh.bdr_attributes])), "boundaries,", str(nv), "nodes,", str(mesh.GetGlobalNE()), "elements")
    material = generateMaterial(fem, geom)
    mfem.print_("Material generated for", str([name for name in material.keys()]))
    models = generateModel(fem, geom, mesh, material)
    mfem.print_("Model generated for", str([m.name for m in models]))
    solvers = generateSolver(fem, mesh, models)
    mfem.print_("Solver generated for", str([s.name for s in solvers]))
    mfem.print_()

    for i, s in enumerate(solvers):
        mfem.print_("------------Solver " + str(i) + ": " + s.name + " started --------------------")
        s.execute()
