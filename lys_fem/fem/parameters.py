import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


class Parameters(dict):
    def getSolved(self):
        if len(self) == 0:
            return {}
        eqs = [sp.Eq(key, item) for key, item in self.items()]
        sol = sp.solve(eqs, list(self.keys()))
        if isinstance(sol, dict):
            return sol
        else:
            return {key: value for key, value in zip(self.keys(), sol[0])}

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = parse_expr(key)
        if isinstance(value, str):
            value = parse_expr(value)
        super().__setitem__(key, value)

    def saveAsDictionary(self):
        return {str(key): str(item) for key, item in self.items()}

    @staticmethod
    def loadFromDictionary(d):
        p = Parameters()
        for key, item in d.items():
            p[parse_expr(key)] = parse_expr(item)
        return p
    

class RandomFields(dict):
    def add(self, name, type, shape=(), tdep=False):
        self[name] = RandomField(type, shape, tdep)

    def saveAsDictionary(self):
        return {key: value.saveAsDictionary() for key, value in self.items()}
    
    @classmethod
    def loadFromDictionary(cls, d):
        res = RandomFields()
        for key, value in d.items():
            res[key] = RandomField.loadFromDictionary(value)
        return res
    

class RandomField:
    def __init__(self, type, shape, tdep):
        self.type = type
        self.tdep = tdep
        self.shape = shape

    def set(self, type, shape, tdep):
        self.type = type
        self.tdep = tdep
        self.shape = shape

    def saveAsDictionary(self):
        return {"type": self.type, "tdep": self.tdep, "shape": self.shape}
    
    @classmethod
    def loadFromDictionary(cls, d):
        return RandomField(d["type"], d.get("shape", ()), d["tdep"])