import numpy as np
from . import util


def generateMaterial(fem, mesh):
    mats = fem.materials.materialDict(mesh.dim)
    mats.update(fem.geometries.geometryParameters())
    result = {}
    for key, value in mats.items():
        if key == "J":
            J = util.coef(value, mesh, default=np.eye(3), name="J")
            result["J"] = J
            result["detJ"] = J.det()
        else:
            result[key] = util.coef(value, mesh, name=key)
    return result
