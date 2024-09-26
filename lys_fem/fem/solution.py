class SolutionFields(dict):
    def add(self, name, path, expression, index=-1):
        self[name] = SolutionField(path, expression)

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
        self._path = path
        self._expression = expression
        self._index = index

    @property
    def solution(self):
        from .solution import FEMSolution
        return FEMSolution(self._path)
    
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

    @property
    def obj(self):
        return self._sol
