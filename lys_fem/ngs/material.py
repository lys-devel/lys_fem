import numpy as np
from . import util


def generateMaterial(fem, mesh):
    mats = fem.materials.materialDict(mesh.dim)
    mats.update(fem.geometries.geometryParameters())
    result = {}
    for key, value in mats.items():
        if key == "J":
            J = util.coef(value, mesh, default=util.generateCoefficient(np.eye(3).tolist()))
            result["J"] = J.Compile()
            result["detJ"] = det(J).Compile()
        else:
            result[key] = util.coef(value, mesh, name=key)
    return result

def det(J):
    return J[0,0]*J[1,1]*J[2,2] + J[0,1]*J[1,2]*J[2,0] + J[0,2]*J[1,0]*J[2,1] - J[0,2]*J[1,1]*J[2,0] - J[0,1]*J[1,0]*J[2,2] - J[0,0]*J[1,2]*J[2,1]
