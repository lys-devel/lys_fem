from . import mfem, coef
import sympy as sp

x, y, z, t, dV, dS = sp.symbols("x,y,z,t,dV,dS")

def grad(v):
    if isinstance(v, sp.Matrix):
        return sp.Matrix.hstack(*[grad(x) for x in v])
    if z in v.free_symbols:
        return sp.Matrix([v.diff("x"), v.diff("y"), v.diff("z")])
    elif y in v.free_symbols:
        return sp.Matrix([v.diff("x"), v.diff("y")])
    else:
        return sp.Matrix([v.diff("x")])


def TrialFunction(name, mesh, ess_bdrs, value, nvar=None, order=1):
    if nvar is not None:
        return sp.Matrix([TrialFunction(name+str(i+1), mesh, ess_bdrs[i], value[i]) for i in range(nvar)])
    args = {1: (x,t), 2: (x,y,t), 3:(x,y,z,t)}
    result = sp.Function("trial_"+name)(*args[mesh.SpaceDimension()])
    if ess_bdrs is not None:
        result.mfem = _MFEMInfo(name, mesh, ess_bdrs, value, order)
    return result


def TestFunction(trial):
    if isinstance(trial, sp.Matrix):
        return sp.Matrix([TestFunction(t) for t in trial])
    result = sp.Function("test_"+trial.func.name.replace("trial_", ""))(*trial.free_symbols)
    if hasattr(trial, "mfem"):
        result.mfem = trial.mfem
    return result


class _MFEMInfo:
    def __init__(self, name, mesh, ess_bdrs, value, order=1):
        self._name = name
        self._fec = mfem.H1_FECollection(order, mesh.Dimension())
        self._space = mfem.FiniteElementSpace(mesh, self._fec, 1)
        self._ess_bdrs = self.__getEssentialBoundary(ess_bdrs)
        self.setValue(value)

    def __getEssentialBoundary(self, ess_bdrs):
        ess_bdr = mfem.intArray(self.space.GetMesh().bdr_attributes.Max())
        ess_bdr.Assign(0)
        for i in ess_bdrs:
            ess_bdr[i - 1] = 1
        return ess_bdr
    
    def setValue(self, coef):
        self._x = mfem.GridFunction(self._space)
        self._x.ProjectCoefficient(coef)

    def dualToPrime(self, d):
        res = mfem.Vector(self.space.GetTrueVSize())
        M = _BilinearForm._bilinearForm(self.space, domainInteg=mfem.MassIntegrator())
        solver, prec = mfem.getSolver("CG", "GS")
        solver.SetOperator(M)
        solver.Mult(d, res)
        return res

    @property
    def dimension(self):
        return self.space.GetMesh().SpaceDimension()

    @property
    def space(self):
        return self._space
    
    @property
    def ess_bdrs(self):
        return self._ess_bdrs
    
    @property
    def x(self):
        return self._x


class WeakformParser:
    def __init__(self, wf, trials, coeffs):
        self._wf = wf
        self._trials = trials
        self._tests = [TestFunction(t) for t in trials]
        self._coeffs = coeffs

        self._bilinearForms = [[_BilinearForm(self._wf, trial, test) for test in self._tests] for trial in self._trials]
        self._linear = sum([sum([b.isNonlinear for b in bs]) for bs in self._bilinearForms])==0
        self._offsets =mfem.intArray([0]+[t.mfem.space.GetTrueVSize() for t in self._trials])
        self._offsets.PartialSum()
        self._init = False

    def initialValue(self):
        self._x = mfem.BlockVector(self._offsets)
        for i, trial in enumerate(self._trials):
            gf = mfem.GridFunction(trial.mfem.space, trial.mfem.x)
            gf.GetTrueDofs(self._x.GetBlock(i))
        return self._x

    def update(self, x):
        # skip if the system is linear
        if self._init and self._linear:
            return self._M, self._K, self._b
        self._init = True

        # update coefficient for nonlinear problem
        x = mfem.BlockVector(x, self._offsets)
        if not self._linear:
            for i, trial in enumerate(self._trials):
                print("set", _replaceFuncs(trial))
                self._coeffs[str(_replaceFuncs(trial))] = coef.VariableCoef(trial, x.GetBlock(i))

        self._b = mfem.BlockVector(self._offsets)
        self._bv = [_LinearForm(self._wf, test).getDofs(self._coeffs) for test in self._tests]

        self._m = [[b.getMassMatrix(self._coeffs) for b in bs] for bs in self._bilinearForms]
        self._k = [[b.getMatrix(self._coeffs, x.GetBlock(i), self._bv[j]) for j, b in enumerate(bs)] for i, bs in enumerate(self._bilinearForms)]

        for j, (test, b) in enumerate(zip(self._tests, self._bv)):
            gf = mfem.GridFunction(test.mfem.space, b)
            gf.GetTrueDofs(self._b.GetBlock(j))

        self._M = mfem.BlockOperator(self._offsets)
        self._K = mfem.BlockOperator(self._offsets)
        for i, trial in enumerate(self._trials):
            for j, test in enumerate(self._tests):
                self._M.SetBlock(i, j, self._m[j][i])
                self._K.SetBlock(i, j, self._k[j][i])

        return self._M, self._K, self._b
        #return mfem.SparseMatrix(self._M.CreateMonolithic()), mfem.SparseMatrix(self._K.CreateMonolithic()), self._x, self._b

class _LinearForm:
    def __init__(self, wf, test):
        self._weakform = wf
        self._test = test

    def getDofs(self, coeffs):
        test, wf = self._test, self._weakform

        # f.dot(∇v) * dV term
        scoef_V1 = self.__sympyCoeff1D(wf.coeff(dV), test, test_deriv=True)
        coef_V1 = SubsCoeff(scoef_V1, coeffs, test.mfem.dimension, "vector")
        integ_V1 = self.__linearIntegrator(coef_V1, True)

        # f * v * dV term
        scoef_V2 = self.__sympyCoeff1D(wf.coeff(dV), test)
        coef_V2 = SubsCoeff(scoef_V2, coeffs, test.mfem.dimension)
        integ_V2 = self.__linearIntegrator(coef_V2, False)

        # f * v * dS term
        scoef_S = self.__sympyCoeff1D(wf.coeff(dS), test)
        coef_S = SubsCoeff(scoef_S, coeffs, test.mfem.dimension)
        integ_S = self.__linearIntegrator(coef_S, boundary=True)
        return self.__linearForm(test.mfem.space, domainInteg=[integ_V2], boundaryInteg=integ_S)

    @classmethod
    def __linearIntegrator(cls, coef, test_deriv=False, boundary=False):
        if boundary:
            return mfem.BoundaryLFIntegrator(coef)
        if test_deriv:
            return mfem.DomainLFGradIntegrator(coef)
        else:
            return mfem.DomainLFIntegrator(coef)

    @classmethod
    def __sympyCoeff1D(cls, wf, test, test_deriv=False):
        wf = _replaceFuncs(wf, True)
        if test_deriv is False:
            return wf.coeff(sp.Symbol(test.func.name))
        else:
            return sp.Matrix([wf.coeff(_replaceFuncs(t)) for t in grad(test)])

    @staticmethod
    def __linearForm(space, domainInteg=None, boundaryInteg=None):
        # initialization
        if domainInteg is None:
            domainInteg = []
        if not hasattr(domainInteg, "__iter__"):
            domainInteg = [domainInteg]
        if boundaryInteg is None:
            boundaryInteg = []
        if not hasattr(boundaryInteg, "__iter__"):
            boundaryInteg = [boundaryInteg]

        # create Linear form of mfem
        b = mfem.LinearForm(space)
        for i in domainInteg:
            b.AddDomainIntegrator(i)
        for i in boundaryInteg:
            b.AddBoundaryIntegrator(i)
        b.Assemble()

        return b


class _BilinearForm:
    def __init__(self, wf, trial, test):
        self._weakform = wf
        self._trial = trial
        self._test = test

        mass_coef = self.__sympyCoeff(self._weakform.coeff(dV), trial.diff(t), test)
        self._mass = _BilinearFormMatrix(trial, test, mass_coef)

        grad2_coef = self.__sympyCoeff(wf.coeff(dV), trial, test, True, True)
        self._grad2 = _BilinearFormMatrix(trial, test, grad2_coef, True, True)

        self.isNonlinear = sum(["trial_" in str(c) for c in [mass_coef, grad2_coef]])

    def getMassMatrix(self, coeffs):
        return self._mass.getMatrix(coeffs)

    def getMatrix(self, coeffs, x=None, b=None):
        return self._grad2.getMatrix(coeffs, x, b)


    @classmethod
    def __sympyCoeff(cls, wf, trial, test, trial_deriv=False, test_deriv=False):
        """
        Get bilinearform coefficient of weakform for trial and test functions.

        For exsample, if wf = c * u * v, c will be returned as sympy symbol.
        If trial_deriv is True, the coefficient is vector: wf = c.dot(∇u) *v.
        If test_deriv is True, the coefficient is vector: wf = u * c.dot(∇u)
        If both trial_ and test_deriv is True, the coefficient is matrix: wf = ∇u.dot(c*∇v)

        args:
            wf(sympy expression): The weakform.
            trial(sympy expression): The trial function
            test(sympy expression): The test function.
            trial_deriv(bool): See above.
            test_driv(bool): See above.
        """
        wf = _replaceFuncs(wf)
        if trial_deriv is False:
            wf_t = cls.__coeff(wf, _replaceFuncs(trial))
            if test_deriv is False:
                return wf_t.coeff(sp.Symbol(test.func.name))
            else:
                return sp.Matrix([wf_t.coeff(_replaceFuncs(t)) for t in grad(test)])
        else:
            res = []
            for tri in grad(trial):
                wf_t = cls.__coeff(wf, _replaceFuncs(tri))
                if test_deriv is False:
                    res.append(wf_t.coeff(_replaceFuncs(test)))
                else:
                    res.append([wf_t.coeff(_replaceFuncs(t)) for t in grad(test)])
            return sp.Matrix(res)

    @classmethod
    def __coeff(cls, expr, x_trial):
        if expr == 0:
            return sp.core.numbers.Zero()
        p = sp.poly(expr, x_trial)
        ac = p.all_coeffs()
        res = sp.core.numbers.Zero()
        for order in range(p.degree()):
            res += ac[order] * x**(p.degree() - order - 1)
        return res


class _BilinearFormMatrix:
    def __init__(self, trial, test, coef_sympy, trial_deriv=False, test_deriv=False):
        self._trial = trial
        self._test = test
        self._coef = coef_sympy
        self._trial_deriv = trial_deriv
        self._test_deriv = test_deriv
        self._dim = test.mfem.dimension

    def getMatrix(self, coeffs, x=None, b=None):
        coef_V = SubsCoeff(self._coef, coeffs, self._dim, self.__checkType())
        integ_V = self.__bilinearIntegrator(coef_V, self._trial_deriv, self._test_deriv)
        return self.__generateMatrix(x,b,domainInteg=integ_V)
    
    def __checkType(self):
        if self._trial_deriv:
            if self._test_deriv:
                return "matrix"
        else:
            if not self._test_deriv:
                return "scalar"
        return "vector"

    def __generateMatrix(self, x=None, b=None, domainInteg=None, boundaryInteg=None):
        if self._trial.mfem.space == self._test.mfem.space:
            return self._bilinearForm(self._trial.mfem.space, self._trial.mfem.ess_bdrs, x, b,
                                      domainInteg=domainInteg, boundaryInteg=boundaryInteg)
        else:
            return self._mixedBilinearForm(self._trial.mfem.space, self._test.mfem.space,
                                           self._trial.mfem.ess_bdrs, self._test.mfem.ess_bdrs, x, b, 
                                           domainInteg=domainInteg, boundaryInteg=boundaryInteg)

    @classmethod
    def __bilinearIntegrator(cls, coef, trial_deriv=False, test_deriv=False):
        if trial_deriv:
            if test_deriv:
                integ= mfem.MixedGradGradIntegrator(coef)
            else:
                integ= mfem.MixedDirectionalDerivativeIntegrator(coef)
        else:
            if test_deriv:
                integ= mfem.MixedScalarWeakDivergenceIntegrator(coef)
            else:
                integ= mfem.MixedScalarMassIntegrator(coef)
        return integ
    
    @staticmethod
    def _bilinearForm(space, ess_bdrs=None, x=None, b=None, domainInteg=None, boundaryInteg=None):
        # initialization
        if domainInteg is None:
            domainInteg = []
        if not hasattr(domainInteg, "__iter__"):
            domainInteg = [domainInteg]
        if boundaryInteg is None:
            boundaryInteg = []
        if not hasattr(boundaryInteg, "__iter__"):
            boundaryInteg = [boundaryInteg]

        # create Bilinear form of mfem
        m = mfem.BilinearForm(space)
        for i in domainInteg:
            m.AddDomainIntegrator(i)
        for i in boundaryInteg:
            m.AddBoundaryIntegrator(i)
        m.Assemble()
        m.Finalize()

        if b is not None:
            m.EliminateEssentialBC(ess_bdrs, x, b)

        result = m.SpMat()
        result._bilin = m
        return result

    @staticmethod
    def _mixedBilinearForm(space1, space2, ess_bdrs1=None, ess_bdrs2=None, x=None, b=None, domainInteg=None, boundaryInteg=None):
        # initialization
        if domainInteg is None:
            domainInteg = []
        if not hasattr(domainInteg, "__iter__"):
            domainInteg = [domainInteg]
        if boundaryInteg is None:
            boundaryInteg = []
        if not hasattr(boundaryInteg, "__iter__"):
            boundaryInteg = [boundaryInteg]

        # create Bilinear form of mfem
        m = mfem.MixedBilinearForm(space1, space2)
        for i in domainInteg:
            m.AddDomainIntegrator(i)
        for i in boundaryInteg:
            m.AddBoundaryIntegrator(i)
        m.Assemble()
        m.Finalize()
    
        if b is not None:
            m.EliminateTrialDofs(ess_bdrs1, x, b)
            m.EliminateTestDofs(ess_bdrs2)

        # set it to matrix
        result = m.SpMat()
        result._bilin = m

        return result


def _replaceFuncs(wf, removeTrial=False):
    """Used from sympyCoeff"""
    for f in wf.atoms(sp.Function):
        if removeTrial and "trial_" in f.func.name:
            enable = 0
        else:
            enable = 1
        wf = wf.subs(f.diff(t).diff(t), sp.Symbol(f.func.name+"_tt")*enable)
        for var in [x,y,z,t]:
            wf = wf.subs(f.diff(var), sp.Symbol(f.func.name+"_"+str(var))*enable)
        wf = wf.subs(f, sp.Symbol(f.func.name)*enable)
    return wf

def SubsCoeff(scoef, coefs, dim, type="scalar"):
    """
    Substitute sympy coef object by coefs dictionary.
    """
    if type=="vector":
        res = coef.VectorCoef.fromScalars([SubsCoeff(scoef[i], coefs, dim) for i in range(scoef.shape[0])])
    elif type=="matrix":
        res = coef.MatrixCoef.fromScalars([[SubsCoeff(scoef[i,j], coefs, dim) for i in range(scoef.shape[0])] for j in range(scoef.shape[1])])
    else:
        args = tuple(scoef.free_symbols)
        res = sp.lambdify(args, scoef)(*[coefs[str(a)] for a in args])
        if not isinstance(res, coef.ScalarCoef):
            res = coef.ScalarCoef(res, dim)
    return res

