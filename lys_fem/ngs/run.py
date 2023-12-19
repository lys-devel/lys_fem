from .mesh import generateMesh
from .util import generateCoefficient
from .models import generateModel
from .solver import generateSolver


def run(fem, run=True):
    print("\n----------------------------NGS started ---------------------------")
    print()

    mesh = generateMesh(fem)
    print("NGS Mesh generated: ", mesh.ne, "elements, ", mesh.nv, "nodes.")
    print("\tDomains:", mesh.GetMaterials())
    print("\tBoundaries:", mesh.GetBoundaries())
    print()

    mats = fem.materials.materialDict(mesh.dim)
    mats = {key: generateCoefficient(value, mesh) for key, value in mats.items()}
    print("NGS Material generated:")
    print("\tParameters:", {key: value.shape if len(value.shape)>0 else "scalar" for key, value in mats.items()})
    print()

    models = generateModel(fem, mesh, mats)
    print("NGS Models generated:")
    for m in models:
        print("\t"+m.name+":", [key for key in m.variableNames])
    print()

    solvers = generateSolver(fem, mesh, models)
    print("NGS Solvers generated:"+str([s.name for s in solvers]))
    print()
 
    if run:
        for i, s in enumerate(solvers):
            print("------------Solver " + str(i) + ": " + s.name + " started --------------------")
            import time
            start = time.time()
            s.execute()
            print(time.time()-start)
    else:
        return mesh, mats, models, solvers

