import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from lys_fem import util


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

    def eval(self):
        return {str(key): util.NGSFunction(value) for key, value in self.getSolved().items()}

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
        self[name] = util.RandomFieldFunction(type, shape, tdep, name=name)

    def update(self):
        for f in self.values():
            if f.isTimeDependent:
                f.update()

    def saveAsDictionary(self):
        return {key: value.to_dict() for key, value in self.items()}
    
    @classmethod
    def loadFromDictionary(cls, d):
        return RandomFields({key: util.RandomFieldFunction.from_dict(value) for key, value in d.items()})
    
