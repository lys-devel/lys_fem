
from . import mfem
from .mesh import generateMesh
from .material import generateMaterial
from .models import generateModel
from .solver import generateSolver


def run(fem, run=True):
    mfem.print_initialize()
    mesh, nv = generateMesh(fem)
    mfem.print_("Mesh generated: ", str(len([1 for _ in mesh.attributes])), "domains,", str(len([1 for _ in mesh.bdr_attributes])), "boundaries,", str(nv), "nodes,", str(mesh.GetGlobalNE()), "elements")
    material = generateMaterial(fem, mesh)
    mfem.print_("Material generated for", str([name for name in material.keys()]))
    models = generateModel(fem, mesh, material)
    mfem.print_("Model generated for", str([m.name for m in models]))
    solvers = generateSolver(fem, mesh, models)
    mfem.print_("Solver generated for", str([s.name for s in solvers]))
    mfem.print_()

    if run:
        for i, s in enumerate(solvers):
            mfem.print_("------------Solver " + str(i) + ": " + s.name + " started --------------------")
            import time
            start = time.time()
            s.execute()
            print(time.time()-start)
    else:
        return mesh, material, models, solvers
