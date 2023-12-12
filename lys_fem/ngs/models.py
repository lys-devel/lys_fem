from ngsolve import BilinearForm, LinearForm, GridFunction, x
from ..fem import DirichletBoundary

modelList = {}


def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return [modelList[m.name](m, mesh, mat) for m in fem.models]


class NGSModel:
    def __init__(self, model):
        self._model = model

    @property
    def variableName(self):
        return self._model.variableName

    @property
    def dirichletCondition(self):
        conditions = [b for b in self._model.boundaryConditions if isinstance(b, DirichletBoundary)]
        bdr_dir = {i: [] for i in range(self._model.variableDimension())}
        for b in conditions:
            for axis, check in enumerate(b.components):
                if check:
                    bdr_dir[axis].extend(b.boundaries.getSelection())
        return bdr_dir



class CompositeModel:
    def __init__(self, mesh, models, type):
        self._mesh = mesh
        self._models = models
        self._type = type

    def __checkNonlinear(self, wf, trials):
        vars = []
        for trial in trials:
            vars.append(trial)
            for gt in grad(trial):
                vars.append(gt)
        p = sp.poly(wf, *vars)
        for order in p.as_dict().keys():
            if sum(order) > 1:
                return True
        return False

    def solve(self, solver, dt=1):
        fes = self.space
        b = BilinearForm(fes)
        b += self.weakform

        l = LinearForm(fes)
        b.Assemble()
        l.Assemble()

        u = GridFunction(fes)
        u.Set(x)
        
        res = l.vec.CreateVector()
        res.data = l.vec - b.mat * u.vec
        u.vec.data += b.mat.Inverse(fes.FreeDofs()) * res
        print(u.vec)

    def __getFromGridFunctions(self):
        x = mfem.BlockVector(self._block_offset)
        for i, trial in enumerate(self.trialFunctions):
            trial.mfem.x.GetTrueDofs(x.GetBlock(i))
        return x
    
    def __setToGridFunctions(self, x):
        x0 = mfem.BlockVector(x, self._block_offset)
        for i, t in enumerate(self.trialFunctions):
            t.mfem.x.SetFromTrueDofs(x0.GetBlock(i))
            print(t, t.mfem.x.GetDataArray())

    def update(self, x):
        # Translate x to grid functions
        x = mfem.BlockVector(x, self._block_offset)
        x_gfs = []
        for i, trial in enumerate(self.trialFunctions):
            x_gf = mfem.GridFunction(trial.mfem.space)
            x_gf.SetFromTrueDofs(x.GetBlock(i))
            x_gfs.append(x_gf)      

        # Parse matrices by updated coefficients
        coeffs = self.__updateCoefficients(x_gfs)
        self.K, self.b, self._J = self._parser.update(x_gfs, self.dt, coeffs)

    def __updateCoefficients(self, x_gfs):
        coeffs = {}
        # prepare coefficient for trial functions and its derivative.
        for gf, trial in zip(x_gfs, self.trialFunctions):
            coeffs[trial.mfem.name] = mfem.generateCoefficient(gf)
            if not trial.mfem.isBoundary:
                for d, gt in enumerate(trial.mfem.gradNames):
                    gfd = mfem.GridFunction(trial.mfem.space)
                    gf.GetDerivative(1, d, gfd)
                    coeffs[gt] = mfem.generateCoefficient(gfd)

        # coefficient for dt and previous value
        if self._update_t:
            coeffs["dt"] = mfem.generateCoefficient(self.dt)
            for trial in self.trialFunctions:
                p = prev(trial)
                coeffs[str(p)] = mfem.generateCoefficient(trial.mfem.x)
            self._update_t = False
        return coeffs

    def __call__(self, x):
        K, b = self.K, self.b
        res = mfem.Vector(x.Size())
        K.Mult(x, res)
        res -= b
        return res

    def grad(self, x):
        return self._J

    @property
    def weakform(self):
        return sum(m.weakform for m in self._models)

    @property
    def space(self):
        spaces = []
        for m in self._models:
            spaces.extend(m.spaces)
        result = spaces[0]
        for sp in spaces[1:]:
            result = result * sp
        return result
        
    @property
    def isNonlinear(self):
        return self._nonlinear

    @property
    def solution(self):
        return {t.mfem.name.replace("trial_", ""): t.mfem.x.getData() for t in self.trialFunctions}

 