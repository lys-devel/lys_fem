import time
import numpy as np
from .mesh import generateMesh
from .util import generateCoefficient
from .models import generateModel
from .solver import generateSolver


def run(fem, run=True):
    print("\n----------------------------NGS started ---------------------------")
    print()

    start = time.time()
    mesh = generateMesh(fem)
    print("NGS Mesh generated in", '{:.2f}'.format(time.time()-start) ,"second : ", mesh.ne, "elements, ", mesh.nv, "nodes.")
    print("\tDomains:", mesh.GetMaterials())
    print("\tBoundaries:", mesh.GetBoundaries())
    print()

    start = time.time()
    mats = generateMaterial(fem, mesh)
    print("NGS Material generated in", '{:.2f}'.format(time.time()-start), ":")
    print("\tParameters:", {key: value.shape if len(value.shape)>0 else "scalar" for key, value in mats.items()})
    print()

    start = time.time()
    model = generateModel(fem, mesh, mats)
    print("NGS Models generated in ", '{:.2f}'.format(time.time()-start), ":")
    for m in model.models:
        print("\t"+m.name+":", [v.name for v in m.variables])
    print()

    start = time.time()
    solvers = generateSolver(fem, mesh, model)
    print("NGS Solvers generated in ", '{:.2f}'.format(time.time()-start), ":"+str([s.name for s in solvers]))
    print()
 
    if run:
        for i, s in enumerate(solvers):
            print("------------Solver " + str(i) + ": " + s.name + " started --------------------")
            start = time.time()
            s.execute()
            print(time.time()-start)
    else:
        return mesh, mats, model, solvers

def generateMaterial(fem, mesh):
    mats = fem.materials.materialDict(mesh.dim)
    mats.update(fem.geometries.geometryParameters())
    result = {}
    for key, value in mats.items():
        if key == "J":
            J = generateCoefficient(value, mesh, default=generateCoefficient(np.eye(3).tolist()))
            result["J"] = J.Compile()
            result["detJ"] = det(J).Compile()
        else:
            result[key] = generateCoefficient(value, mesh)
    return result

def det(J):
    return J[0,0]*J[1,1]*J[2,2] + J[0,1]*J[1,2]*J[2,0] + J[0,2]*J[1,0]*J[2,1] - J[0,2]*J[1,1]*J[2,0] - J[0,1]*J[1,0]*J[2,2] - J[0,0]*J[1,2]*J[2,1]
