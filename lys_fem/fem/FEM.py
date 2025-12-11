from lys_fem import util

from .base import FEMObjectList
from .parameters import Parameters, RandomFields
from .geometry import GeometryGenerator
from .mesh import OccMesher
from .material import Material, Materials
from .model import loadModel
from .solver import FEMSolver
from .solution import SolutionFields


class FEMProject:
    def __init__(self):
        self.reset()

    def reset(self):
        self._parallel = False
        self._params = Parameters()
        self._geom = GeometryGenerator()
        self._geom.setParent(self)
        self._mesher = OccMesher(self)
        self._materials = Materials(self, [Material(objName="Material1")])
        self._models = FEMObjectList(self)
        self._solvers = FEMObjectList(self)
        self._solutions = SolutionFields()
        self._randoms = RandomFields()
        self._submit = {"nthreads": 4}

    def saveAsDictionary(self, parallel=False):
        d = {"parallel": parallel}
        d["parameters"] = self._params.saveAsDictionary()
        d["geometries"] = self._geom.saveAsDictionary()
        d["mesh"] = self._mesher.saveAsDictionary()
        d["materials"] = [m.saveAsDictionary() for m in self._materials]
        d["models"] = [m.saveAsDictionary() for m in self._models]
        d["solvers"] = [s.saveAsDictionary() for s in self._solvers]
        d["solutionFields"] = self._solutions.saveAsDictionary()
        d["randomFields"] = self._randoms.saveAsDictionary()
        d["submit"] = self._submit
        return d

    def loadFromDictionary(self, d):
        self._parallel = d.get("parallel", False)
        if "parameters" in d:
            self._params = Parameters.loadFromDictionary(d["parameters"])
        if "geometries" in d:
            self._geom = GeometryGenerator.loadFromDictionary(d["geometries"])
            self._geom.setParent(self)
        if "mesh" in d:
            self._mesher = OccMesher.loadFromDictionary(d["mesh"])
            self._mesher.setParent(self)
        if "materials" in d:
            self._materials = Materials(self, [Material.loadFromDictionary(dic) for dic in d["materials"]])
        if "models" in d:
            self._models = FEMObjectList(self, [loadModel(dic) for dic in d["models"]])
        if "solvers" in d:
            self._solvers = FEMObjectList(self, [FEMSolver.loadFromDictionary(dic) for dic in d["solvers"]])
        if "solutionFields" in d:
            self._solutions = SolutionFields.loadFromDictionary(d["solutionFields"])
        if "randomFields" in d:
            self._randoms = RandomFields.loadFromDictionary(d["randomFields"])
        if "submit" in d:
            self._submit = d["submit"]

    @classmethod
    def fromFile(cls, file):
        res = FEMProject()
        with open(file) as f:
            d = eval(f.read())
        res.loadFromDictionary(d)
        return res

    @property
    def dimension(self):
        return self.geometries.generateGeometry().dimension

    @property
    def parallel(self):
        return self._parallel
    
    @property
    def parameters(self):
        return self._params

    @property
    def geometries(self):
        return self._geom

    @property
    def mesher(self):
        return self._mesher

    @property
    def materials(self):
        return self._materials

    @property
    def models(self):
        return self._models

    @property
    def solvers(self):
        return self._solvers

    @property
    def submitSetting(self):
        return self._submit

    @property
    def solutionFields(self):
        return self._solutions
    
    @property
    def randomFields(self):
        return self._randoms

    def evaluator(self, solutions=True):
        class FEMParameters(dict):
            def __getitem__(self, expr):
                return util.eval(expr, dict(self), name=str(expr))

        d = FEMParameters(util.consts.asdict())
        d.update(self.parameters.eval())
        if solutions:
            d.update(self.solutionFields.eval())
        d.update(self.randomFields)
        d.update(self.materials.eval(d))
        d.update(self.geometries.geometryParameters())
        d.update({v.name: util.TrialFunction(v) for v in self.compositeModel.variables})
        return d

    @property
    def compositeModel(self):
        return CompositeModel(self._models)

    @property
    def mesh(self):
        geom = self.geometries.generateGeometry()
        mesh = self.mesher.generate(geom)
        return util.Mesh(mesh)

    @property
    def domainAttributes(self):
        return self.geometries.geometryAttributes(self.dimension)

    @property
    def boundaryAttributes(self):
        return self.geometries.geometryAttributes(self.dimension-1)

    def getMeshWave(self, dim=None, nomesh=False):
        if dim is None:
            dim = self.dimension
        if nomesh:
            mesher = OccMesher(self)
        else:
            mesher = self._mesher
        geom = self._geom.generateGeometry()
        return mesher.generate(geom).getMeshWave(dim=dim)

    def run(self):
        from .run import run
        run(self)


class CompositeModel:
    def __init__(self, models):
        self._models = models

    def weakforms(self, mat):
        tnt = {v.name: (util.trial(v), util.test(v)) for v in self.variables}
        return sum([model.weakform(tnt, mat) for model in self._models])
    
    def discretize(self):
        d = {}
        for m in self._models:
            d.update(m.discretize(util.dti))
        return d

    def updater(self):
        d = {}
        for m in self._models:
            d.update(m.updater(util.dti))
        return d

    def initialValue(self, fes, mat):
        x0 = sum([m.initialValues(mat) for m in self._models], [])
        v0 = sum([m.initialVelocities(mat) for m in self._models], [])
        x = fes.gridFunction(x0)
        v = fes.gridFunction(v0)
        a = fes.gridFunction()
        return x, v, a
    
    @property
    def models(self):
        return self._models
    
    @property
    def variables(self):
        return sum([m.functionSpaces() for m in self._models], [])      