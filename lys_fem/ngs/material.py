from . import util


def generateMaterial(fem, solutions=True):
    d = fem.evaluator(solutions=solutions)
    return NGSParams(fem, d)


class NGSParams(dict):
    def __init__(self, fem, sols):
        super().__init__(sols)
        self._fem = fem

    def __getitem__(self, expr):
        return util.eval(expr, dict(self), name=str(expr))
    
    def updateSolutionFields(self, step):
        for key, f in self.items():
            if isinstance(f, util.SolutionFieldFunction) and f.isTimeDependent:
                self._fem.solutionFields[key].update(step)
            if isinstance(f, util.RandomFieldFunction) and f.isTimeDependent:
                f.update()
