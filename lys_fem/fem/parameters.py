import sympy as sp

class Scaling:
    def __init__(self, length=1, time=None, mass=None, current=None, temperature=None, amount=None, luminous=None):
        self.set(length, time, mass, current, temperature, amount, luminous)

    def set(self, length=1, time=None, mass=None, current=None, temperature=None, amount=None, luminous=None):
        if time is None:
            time = length
        if mass is None:
            mass = length**3
        if current is None:
            current = 1
        if temperature is None:
            temperature = 1
        if amount is None:
            amount = 1
        if luminous is None:
            luminous = 1
        self._norms = [length, time, mass, current, temperature, amount, luminous]

    def getScaling(self, unit=None):
        if unit is not None:
            m, s, kg, A, K, mol, cd = self.__parseScale(unit)
        result = 1
        for unit, order in zip(self._norms, [m,s,kg,A,K,mol,cd]):
            result *= unit**order
        return result

    def __parseScale(self, unit):
        from astropy import units
        u = units.Unit(unit).decompose()
        #scale = self._unit.scale
        m, s, kg, A, K, mol, cd = 0, 0, 0, 0, 0, 0, 0
        for base, power in zip(u.bases, u.powers):
            if base == units.m:
                m += power
            elif base == units.s:
                s += power
            elif base == units.kg:
                kg += power
            elif base == units.A:
                A += power
            elif base == units.K:
                K += power
            elif base == units.mol:
                mol += power
            elif base == units.cd:
                cd += power
        return m, s, kg, A, K, mol, cd 


class Parameters(dict):
    def getSolved(self):
        eqs = [sp.Eq(key, item) for key, item in self.items()]
        sol = sp.solve(eqs, list(self.keys()))
        if isinstance(sol, dict):
            return sol
        else:
            return {key: value for key, value in zip(self.keys(), sol[0])}
