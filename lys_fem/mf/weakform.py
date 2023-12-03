from . import mfem, coef
import sympy as sp
import numpy as np

x, y, z, t, dV, dS = sp.symbols("x,y,z,t,dV,dS")

def grad(v):
    if isinstance(v, sp.Matrix):
        return sp.Matrix.hstack(*[grad(x) for x in v]).T
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

    @property
    def isNonlinear(self):
        return not self._linear

    def initialValue(self):
        self._x = mfem.BlockVector(self._offsets)
        for i, trial in enumerate(self._trials):
            gf = mfem.GridFunction(trial.mfem.space, trial.mfem.x)
            gf.GetTrueDofs(self._x.GetBlock(i))
        return self._x

    def update(self, x):
        # skip if the system is linear
        if self._init and self._linear:
            return self._M, self._K, self._b, self._M, self._K
        self._init = True

        # update coefficient for nonlinear problem
        x = mfem.BlockVector(x, self._offsets)
        x_gfs = []
        for i, trial in enumerate(self._trials):
            x_gf = mfem.GridFunction(trial.mfem.space)
            x_gf.SetFromTrueDofs(x.GetBlock(i))
            x_gfs.append(x_gf)      

        # prepare coefficient for trial functions and its derivative.
        if not self._linear:
            for gf in x_gfs:
                self._coeffs[str(_replaceFuncs(trial))] = coef.generateCoefficient(gf, trial.mfem.dimension)
                for d in range(trial.mfem.dimension):
                    xyz = sp.symbols("x,y,z")[d]
                    gfd = mfem.GridFunction(trial.mfem.space)
                    gf.GetDerivative(1, d, gfd)
                    self._coeffs[str(_replaceFuncs(trial.diff(xyz)))] = coef.generateCoefficient(gfd, trial.mfem.dimension)
                
        self.__updateResidual(x_gfs)
        if self._linear:
            return self._M, self._K, self._b, self._M, self._K
        else:
            self.__updateJacobian(x)
            return self._M, self._K, self._b, self._gM, self._gK
        #return mfem.SparseMatrix(self._M.CreateMonolithic()), mfem.SparseMatrix(self._K.CreateMonolithic()), self._x, self._b

    def __updateResidual(self, x):
        """Calculate M, K, b"""
        self._b = mfem.BlockVector(self._offsets)
        self._bv = [_LinearForm(self._wf, test).getDofs(self._coeffs) for test in self._tests]

        self._m = [[b.getMassMatrix(self._coeffs) for b in bs] for bs in self._bilinearForms]
        self._k = [[b.getMatrix(self._coeffs, x[i], self._bv[j]) for j, b in enumerate(bs)] for i, bs in enumerate(self._bilinearForms)]

        for j, (test, b) in enumerate(zip(self._tests, self._bv)):
            gf = mfem.GridFunction(test.mfem.space, b)
            gf.GetTrueDofs(self._b.GetBlock(j))

        self._M = mfem.BlockOperator(self._offsets)
        self._K = mfem.BlockOperator(self._offsets)
        for i, trial in enumerate(self._trials):
            for j, test in enumerate(self._tests):
                self._M.SetBlock(i, j, self._m[j][i])
                self._K.SetBlock(i, j, self._k[j][i])

    def __updateJacobian(self, x):
        self._gm = [[b.getMassMatrixJacobian(self._coeffs) for b in bs] for bs in self._bilinearForms]
        self._gk = [[b.getMatrixJacobian(self._coeffs) for b in bs] for bs in self._bilinearForms]

        self._gM = mfem.BlockOperator(self._offsets)
        self._gK = mfem.BlockOperator(self._offsets)
        for i, trial in enumerate(self._trials):
            for j, test in enumerate(self._tests):
                self._gM.SetBlock(i, j, self._gm[j][i])
                self._gK.SetBlock(i, j, self._gk[j][i])

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

        mass_coef = self._getCoeffs(wf.coeff(dV), trial.diff(t), test)
        self._mass = _BilinearFormMatrix(trial, test, mass_coef)

        mass_coef_jac = self._getJacobiCoeffs(self._weakform.coeff(dV), trial.diff(t), test)
        self._mass_jac = _BilinearFormMatrix(trial, test, mass_coef_jac)

        stiff_coef = self._getCoeffs(wf.coeff(dV).subs(trial.diff(t), 0), trial, test)
        self._stiff = _BilinearFormMatrix(trial, test, stiff_coef)

        stiff_coef_jac = self._getJacobiCoeffs(self._weakform.coeff(dV), trial, test)
        self._stiff_jac = _BilinearFormMatrix(trial, test, stiff_coef_jac)

        self.isNonlinear = sum(["trial_" in str(c) for c in mass_coef + stiff_coef]) != 0

    def getMassMatrix(self, coeffs):
        return self._mass.getMatrix(coeffs)
    
    def getMassMatrixJacobian(self, coeffs):
        return self._mass_jac.getMatrix(coeffs)

    def getMatrix(self, coeffs, x=None, b=None):
        return self._stiff.getMatrix(coeffs, x, b)
    
    def getMatrixJacobian(self, coeffs):
        return self._stiff_jac.getMatrix(coeffs)

    @classmethod
    def _getCoeffs(cls, wf, trial, test):
        c0, v0 = list(cls.__getCoeffs_single(wf.diff(test), trial))
        c1, v1=[],[]
        for t in grad(test):
            tmp = cls.__getCoeffs_single(wf.diff(t), trial)
            c1.append(tmp[0])
            v1.append(tmp[1])
        return c0, v0, sp.Matrix(c1), sp.Matrix.hstack(*v1).T # coef for u*v, ∇u*v, u*∇v, ∇u*∇v

    @classmethod
    def __getCoeffs_single(cls, wf, trial):
        dim = _dimension(trial)
        diffs = [trial.diff(x), trial.diff(y), trial.diff(z)][:dim]
        p = sp.poly(wf, trial, *diffs)

        def diff_arg(orders):
            res = 1
            for d, o in zip(diffs, orders[1:]):
                res *= d**o
            return res
        
        def diff_arg_vec(orders):
            res = [0] * dim
            for di in range(1, dim+1):
                if orders[di] != 0:
                    ord = list(orders)
                    ord[di] -=1
                    res[di-1] = diff_arg(ord)
            return sp.Matrix(res)

        c, v = 0, sp.Matrix([0]*dim)
        for args, value in p.as_dict().items():
            #if args[0] == 0:
            if sum(args[1:]) != 0:
                v += value / sum(args[1:]) * trial**args[0] * diff_arg_vec(args)
            else:
                if args[0] != 0:
                    c += value * trial**(args[0]-1) * diff_arg(args)
        return c, v

    @classmethod
    def _getJacobiCoeffs(cls, wf, trial, test):
        dim = _dimension(trial)
        diffs_trial = [trial.diff(x), trial.diff(y), trial.diff(z)][:dim]
        diffs_test = [test.diff(x), test.diff(y), test.diff(z)][:dim]
        c1 = wf.diff(trial).diff(test)
        c2 = sp.Matrix([wf.diff(t).diff(test) for t in diffs_trial])
        c3 = sp.Matrix([wf.diff(trial).diff(t) for t in diffs_test])
        c4 = sp.Matrix([[wf.diff(t).diff(t2) for t in diffs_trial] for t2 in diffs_test])
        return c1, c2, c3, c4


class _BilinearFormMatrix:
    def __init__(self, trial, test, coeffs):
        self._trial = trial
        self._test = test
        self._coef = coeffs
        self._dim = test.mfem.dimension
        
    def getMatrix(self, coeffs, x=None, b=None):
        def is_zero(val):
            if isinstance(val, sp.Matrix):
                return np.array([is_zero(v) for v in val]).all()
            else:
                return val==0
        integ = []
        if not is_zero(self._coef[0]):
            coef_V0 = SubsCoeff(self._coef[0], coeffs, self._dim, "scalar")
            integ.append(mfem.MixedScalarMassIntegrator(coef_V0))
        if not is_zero(self._coef[1]):
            coef_V1 = SubsCoeff(self._coef[1], coeffs, self._dim, "vector")
            integ.append(mfem.MixedDirectionalDerivativeIntegrator(coef_V1))
        if not is_zero(self._coef[2]):
            coef_V2 = SubsCoeff(self._coef[2], coeffs, self._dim, "vector")
            integ.append(mfem.MixedScalarWeakDivergenceIntegrator(mfem.ScalarVectorProductCoefficient(-1.0, coef_V2)))
        if not is_zero(self._coef[3]):
            coef_V3 = SubsCoeff(self._coef[3], coeffs, self._dim, "matrix")
            integ.append(mfem.MixedGradGradIntegrator(coef_V3))
        return self.__generateMatrix(x,b,domainInteg=integ)

    def __generateMatrix(self, x=None, b=None, domainInteg=None, boundaryInteg=None):
        if self._trial.mfem.space == self._test.mfem.space:
            return self._bilinearForm(self._trial.mfem.space, self._trial.mfem.ess_bdrs, x, b,
                                      domainInteg=domainInteg, boundaryInteg=boundaryInteg)
        else:
            return self._mixedBilinearForm(self._trial.mfem.space, self._test.mfem.space,
                                           self._trial.mfem.ess_bdrs, self._test.mfem.ess_bdrs, x, b, 
                                           domainInteg=domainInteg, boundaryInteg=boundaryInteg)

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

        if b is not None:
            m.EliminateEssentialBC(ess_bdrs, x, b)
        else:
            m.EliminateEssentialBC(ess_bdrs)

        m.Finalize()
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
    
        if b is None:
            x = mfem.Vector(space1.GetVSize())
            b = mfem.Vector(space2.GetVSize())
        m.EliminateTrialDofs(ess_bdrs1, x, b)
        m.EliminateTestDofs(ess_bdrs2)

        # set it to matrix
        m.Finalize()
        result = m.SpMat()
        result._bilin = m

        return result

def _dimension(trial):
    if z in trial.free_symbols:
        return 3
    elif y in trial.free_symbols:
        return 2
    else:
        return 1

def _replaceFuncs(wf, removeTrial=False):
    """Used from sympyCoeff"""
    if not isinstance(wf, (sp.Basic, sp.Matrix)):
        return wf
    for f in wf.atoms(sp.Function):
        if removeTrial and "trial_" in f.func.name:
            enable = 0
        else:
            enable = 1
        wf = wf.subs(f.diff(t).diff(t), sp.Symbol(f.func.name+"_tt")*enable)
        wf = wf.subs(f.diff(t), sp.Symbol(f.func.name+"_t")*enable)
        for var in [x,y,z]:
            wf = wf.subs(f.diff(var), sp.Symbol(f.func.name+"_"+str(var))*enable)
        wf = wf.subs(f, sp.Symbol(f.func.name)*enable)
    return wf

def SubsCoeff(scoef, coefs, dim, type="scalar"):
    """
    Substitute sympy coef object by coefs dictionary.
    """
    if type=="vector":
        return coef.VectorCoef.fromScalars([SubsCoeff(s, coefs, dim) for s in scoef])
    elif type=="matrix":
        return coef.MatrixCoef.fromScalars([[SubsCoeff(scoef[i,j], coefs, dim) for i in range(scoef.shape[0])] for j in range(scoef.shape[1])])
    else:
        if not isinstance(scoef, (sp.Basic, sp.Matrix)):
            return coef.generateCoefficient(scoef, dim)
        scoef = _replaceFuncs(scoef)
        args = tuple(scoef.free_symbols)
        res = sp.lambdify(args, scoef)(*[coefs[str(a)] for a in args])
        if not isinstance(res, coef.ScalarCoef):
            res = coef.generateCoefficient(res, dim)
    return res

