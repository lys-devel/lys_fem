import ngsolve
from . import util


def generateMaterial(fem, mesh):
    mats = fem.materials.materialDict(mesh.dim)
    mats.update(fem.geometries.geometryParameters())
    #mats.update(fem.parameters)
    return NGSParams(mats, mesh, fem.geometries.scale)


class NGSParams(dict):
    def __init__(self, dic, mesh, scale):
        super().__init__()
        self["x"] = ngsolve.x/scale
        self["y"] = ngsolve.y/scale
        self["z"] = ngsolve.z/scale
        self.__parse(dic, mesh)

    def __parse(self, dic, mesh):
        # Replace all items by dic
        while True:
            n = len(self)
            for key, value in dic.items():
                if value.isSympy():
                    value.subs(self)
                if not value.isSympy() and key not in self:
                    self[key] = util.generateCoefficient(value, mesh)
            if len(self) == n:
                break
        # Convert all coefs into NGSFunction
        for key, value in self.items():
            self[key] = util.NGSFunction(self[key], name=key)