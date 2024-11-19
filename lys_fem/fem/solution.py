class SolutionFields(dict):
    def add(self, name, path, expression, index=-1):
        self[name] = SolutionField(path, expression, index)

    def saveAsDictionary(self):
        return {key: value.saveAsDictionary() for key, value in self.items()}

    @classmethod
    def loadFromDictionary(cls, d):
        res = SolutionFields()
        for key, value in d.items():
            res[key] = SolutionField.loadFromDictionary(value)
        return res


class SolutionField:
    def __init__(self, path="", expression="", index=-1):
        self.set(path, expression, index)

    def set(self, path, expression, index):
        self._path = path
        self._expression = expression
        self._index = index
        self._sol = None

    def get(self):
        if self._index is None:
            return self.solution.obj.coef(self.expression, -1)
        else:
            return self.solution.obj.coef(self.expression, self.index)

    def update(self, step):
        self.solution.obj.update(step)
    
    @property
    def solution(self):
        if self._sol is None:
            self._sol = FEMSolution(self._path)
        return self._sol

    @property
    def path(self):
        return self._path
    
    @property
    def expression(self):
        return self._expression
    
    @property
    def index(self):
        return self._index
    
    def saveAsDictionary(self):
        return {"path": self._path, "expression": self._expression, "index": self._index}
    
    @classmethod
    def loadFromDictionary(cls, d):
        return SolutionField(**d)


class FEMSolution:
    _keys = ["point", "line", "triangle", "quad", "tetra", "hexa", "prism", "pyramid"]

    def __init__(self, path=".", solver="Solver0"):
        from .FEM import FEMProject
        from lys_fem.ngs import NGSSolution
        self._fem = FEMProject.fromFile(path + "/input.dic")
        self._path = path
        self._sol = NGSSolution(self._fem, path+"/Solutions/"+solver)

    def eval(self, varName, data_number=0, coords=None):
        return self._sol.eval(varName, data_number, coords)
    
    def integrate(self, expr, data_number=0):
        return self._sol.integrate(expr, data_number)

    @property
    def obj(self):
        return self._sol
