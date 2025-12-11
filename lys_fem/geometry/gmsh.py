import gmsh
import numpy as np
import sympy as sp


gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)


class GmshGeometry:
    _index = 0

    def __init__(self, orders, params={}, scale="auto"):
        self._orders = list(orders)
        self._params = dict(params)
        GmshGeometry._index += 1
        self._name = "Geometry" + str(GmshGeometry._index)
        self.scale = self.__scale(scale)
        tg = _TransGeom(self.scale, params)
        self._model, self._dim = self.__update(self._name, orders, tg)
        self._geom_params = self.__geometryParameters(tg)

    def __del__(self):
        self._model.setCurrent(self._name)
        self._model.remove()

    def __update(self, name, orders, scale):
        model = gmsh.model()
        model.add(name)
        model.setCurrent(name)

        for order in orders:
            order.execute(model, scale)
        model.occ.removeAllDuplicates()
        model.occ.synchronize()
        for i in [1,2,3]:
            if len(model.getEntities(i))!=0:
                dim = i
        for i, obj in enumerate(model.getEntities(3)):
            model.add_physical_group(dim=3, tags=[obj[1]], tag=i + 1)
            model.setPhysicalName(dim=3, tag=i+1, name="domain"+str(i+1))
        for i, obj in enumerate(model.getEntities(2)):
            model.add_physical_group(dim=2, tags=[obj[1]], tag=i + 1)
            if dim == 2:
                model.setPhysicalName(dim=2, tag=i+1, name="domain"+str(i+1))
            elif dim == 3:
                model.setPhysicalName(dim=2, tag=i+1, name="boundary"+str(i+1))
        for i, obj in enumerate(model.getEntities(1)):
            model.add_physical_group(dim=1, tags=[obj[1]], tag=i + 1)
            if dim == 1:
                model.setPhysicalName(dim=1, tag=i+1, name="domain" + str(i+1))
            elif dim == 2:
                model.setPhysicalName(dim=1, tag=i+1, name="boundary" + str(i+1))
            else:
                model.setPhysicalName(dim=1, tag=i+1, name="edge" + str(i+1))
        for i, obj in enumerate(model.getEntities(0)):
            model.add_physical_group(dim=0, tags=[obj[1]], tag=i + 1)
            if dim == 1:
                model.setPhysicalName(dim=0, tag=i+1, name="boundary" + str(i+1))
            else:
                model.setPhysicalName(dim=0, tag=i+1, name="point" + str(i+1))
        return model, dim

    def __geometryParameters(self, scale):
        self._model.setCurrent(self._name)
        result = {}
        for c in self._orders:
            result.update(c.generateParameters(self._model, scale))
        return result

    def __scale(self, scale):
        if scale == "auto":
            def flatten(x):
                for item in x:
                    if hasattr(item, "__iter__"):
                        yield from flatten(item)
                    else:
                        yield abs(item)
            args = flatten([cc for c in self._orders for cc in c.args])
            return min([arg for arg in args if arg!=0])
        else:
            return scale

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        self._model.setCurrent(self._name)
        return self._model

    @property
    def mesh(self):
        self._model.setCurrent(self._name)
        return self._model.mesh

    @property
    def dimension(self):
        return self._dim

    def geometryParameters(self):
        return self._geom_params

    def geometryAttributes(self, dim):
        self._model.setCurrent(self._name)
        return [tag for d, tag  in self._model.getPhysicalGroups(dim)]

    def duplicate(self):
        return GmshGeometry(self._orders, params=self._params)

    def export(self, file):
        self._model.setCurrent(self._name)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(file)


class _TransGeom:
    def __init__(self, scale, params):
        self._scale = scale
        self._params = params

    def __call__(self, value, unit="1"):
        if unit == "m":
            scale = self._scale
        else:
            scale = 1
        if isinstance(value, (list, tuple, np.ndarray)):
            return [self(v, unit) for v in value]
        elif isinstance(value, (float, int, sp.Float, sp.Integer)):
            return value/scale
        else:
            return value.subs(self._params)/scale
