import numpy as np
from ngsolve import x,y,z
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

    mats = generateMaterial(fem, mesh)
    print("NGS Material generated:")
    print("\tParameters:", {key: value.shape if len(value.shape)>0 else "scalar" for key, value in mats.items()})
    print()

    model = generateModel(fem, mesh, mats)
    print("NGS Models generated:")
    for m in model.models:
        print("\t"+m.name+":", [key for key in m.variableNames])
    print()

    solvers = generateSolver(fem, mesh, model)
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
        return mesh, mats, model, solvers

def generateMaterial(fem, mesh):
    mats = fem.materials.materialDict(mesh.dim)
    result = {}
    for key, value in mats.items():
        if key == "CoordsTransform":
            default = generateCoefficient([x,y,z])
            XYZ = generateCoefficient(value, mesh, default=default)
            result["J_T"] = generateCoefficient(np.array([[XYZ[i].Diff(x), XYZ[i].Diff(y), XYZ[i].Diff(z)] for i in range(3)]).T.tolist())
        else:
            result[key] = generateCoefficient(value, mesh)
    return result
