from lys_fem.fem import FEMCoefficient
from . import util


def generateMaterial(fem, solutions=True):
    d = fem.evaluator(solutions=solutions)
    res = NGSParams(fem, d)
    return res


class NGSParams(dict):
    def __init__(self, fem, sols):
        super().__init__(sols)
        self._fem = fem

    def __getitem__(self, expr):
        return _generateCoefficient(expr, dic=dict(self), name=str(expr))
    
    @property
    def jacobi(self):
        res = {}
        if "J" in self:
            res["J"] = self["J"].T
        if "R" in self:
            res["R"] = self["R"]
        return res

    def updateSolutionFields(self, step):
        for key, f in self.items():
            if isinstance(f, util.SolutionFieldFunction) and f.isTimeDependent:
                self._fem.solutionFields[key].update(step)
            if isinstance(f, util.RandomFieldFunction) and f.isTimeDependent:
                f.update()


def _generateCoefficient(coef, name="Undefined", dic={}, J=None):
    if isinstance(coef, FEMCoefficient):
        geom = coef.geometryType.lower()
        if geom == "const":
            return _generateCoefficient(coef.value, name=name, dic=dic,J=J)
        coefs = {key: value for key, value in coef.value.items() if key != "default"}
        if coef.default is not None:
            coefs["default"] = coef.default
        return util.eval(coefs, dic, geom=geom, J=J, name=name)
    else:
        return util.eval(coef, dic, name=name)
