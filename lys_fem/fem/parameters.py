import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from astropy import units

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

    def saveAsDictionary(self):
        return {str(key): str(item) for key, item in self.items()}

    @staticmethod
    def loadFromDictionary(d):
        p = Parameters()
        for key, item in d.items():
            p[parse_expr(key)] = parse_expr(item)
        return p