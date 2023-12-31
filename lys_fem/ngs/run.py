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
            if mesh.dim == 3:
                default = generateCoefficient([x,y,z])
                XYZ = generateCoefficient(value, mesh, default=default)
                result["x"] = x
                result["y"] = y
                result["z"] = z
                result["X"] = XYZ[0]
                result["Y"] = XYZ[1]
                result["Z"] = XYZ[2]
                J = generateCoefficient([[XYZ[i].Diff(x), XYZ[i].Diff(y), XYZ[i].Diff(z)] for i in range(3)])
                result["J_T"] = J
                result["detJ_T"] = J[0,0]*J[1,1]*J[2,2] + J[0,1]*J[1,2]*J[2,0] + J[0,2]*J[1,0]*J[2,1] - J[0,2]*J[1,1]*J[2,0] - J[0,1]*J[1,0]*J[2,2] - J[0,0]*J[1,2]*J[2,1]
            elif mesh.dim == 2:
                default = generateCoefficient([x,y])
                XY = generateCoefficient(value, mesh, default=default)
                result["x"] = x
                result["y"] = y
                result["X"] = XY[0]
                result["Y"] = XY[1]
                J = generateCoefficient([[XY[i].Diff(x), XY[i].Diff(y)] for i in range(2)])
                result["J_T"] = J
                result["detJ_T"] = J[0,0]*J[1,1]  - J[0,1]*J[1,0]
        elif key == "J":
            J = generateCoefficient(value, mesh, default=generateCoefficient(np.eye(3).tolist()))
            J = inv(J).Compile()
            result["J"] = J
            result["detJ"] = det(J).Compile()
        else:
            result[key] = generateCoefficient(value, mesh)
    return result

def det(J):
    return J[0,0]*J[1,1]*J[2,2] + J[0,1]*J[1,2]*J[2,0] + J[0,2]*J[1,0]*J[2,1] - J[0,2]*J[1,1]*J[2,0] - J[0,1]*J[1,0]*J[2,2] - J[0,0]*J[1,2]*J[2,1]

def inv(J):
    mat = [[J[1,1]*J[2,2]-J[1,2]*J[2,1], J[0,2]*J[2,1]-J[0,1]*J[2,2], J[0,1]*J[1,2]-J[0,2]*J[1,1]],
           [J[1,2]*J[2,0]-J[1,0]*J[2,2], J[0,0]*J[2,2]-J[0,2]*J[2,0], J[0,2]*J[1,0]-J[0,0]*J[1,2]],
           [J[1,0]*J[2,1]-J[1,1]*J[2,0], J[0,1]*J[2,0]-J[0,0]*J[2,1], J[0,0]*J[1,1]-J[0,1]*J[1,0]]]
    return generateCoefficient(mat)/det(J)