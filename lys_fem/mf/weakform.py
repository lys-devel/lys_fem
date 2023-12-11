from . import mfem
import sympy as sp
import numpy as np

x, y, z, t, dt = sp.symbols("x,y,z,t,dt")

def grad(v):
    if isinstance(v, sp.Matrix):
        return sp.Matrix.hstack(*[grad(x) for x in v]).T
    if z in v.free_symbols:
        return sp.Matrix([v.diff("x"), v.diff("y"), v.diff("z")])
    elif y in v.free_symbols:
        return sp.Matrix([v.diff("x"), v.diff("y")])
    else:
        return sp.Matrix([v.diff("x")])


def prev(trial, n=1):
    return sp.Symbol("prev_"+trial.func.name.replace("trial_", "")+"_"+str(n))


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
        self._mesh = mesh
        self._dim = mesh.Dimension()
        self._fec = mfem.H1_FECollection(order, self._dim)
        self._space = mfem.FiniteElementSpace(mesh, self._fec, 1)
        self._ess_bdrs = self.__getEssentialBoundary(mesh, ess_bdrs)
        self.setValue(value)

    def __getEssentialBoundary(self, mesh, ess_bdrs):
        ess_bdr = mfem.intArray(mesh.bdr_attributes)
        ess_bdr.Assign(0)
        for i in ess_bdrs:
            ess_bdr[i - 1] = 1
        return ess_bdr
    
    def setValue(self, coef):
        self._x = mfem.GridFunction(self._space)
        self._x.ProjectCoefficient(coef)

    def dualToPrime(self, d):
        self._res = mfem.Vector(self.space.GetTrueVSize())
        self._m = mfem.BilinearForm(self.space)
        self._m.AddDomainIntegrator(mfem.MassIntegrator())
        self._m.Assemble()
        self._m.Finalize()
        self._M = self._m.SpMat()

        self._solver, self._prec = mfem.CGSolver()
        self._solver.SetOperator(self._M)
        self._solver.Mult(d, self._res)
        return self._res

    @property
    def isBoundary(self):
        if isinstance(self._mesh, mfem.SubMesh):
            if self._mesh.GetFrom() == mfem.SubMesh.Boundary:
                return True
        return False

    @property
    def name(self):
        return "trial_" + self._name

    @property    
    def gradNames(self):
        return ["trial_" + self._name + "_" + "xyz"[d] for d in range(self._dim)]

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

def IntegralSymbol(name, type, attr=None):
    symbol = sp.Function(name)()
    symbol._type = type
    symbol._attr = attr
    return symbol

dV = IntegralSymbol("dV", "Volume")
dS = IntegralSymbol("dS", "Boundary")

class WeakformParser:
    def __init__(self, wf, trials, coeffs, integrals):
        self._wf = wf
        self._trials = trials
        self._tests = [TestFunction(t) for t in trials]
        self._coeffs = coeffs
        self._integrals = integrals

        self._wf, self._wf_b = self.__divideTerm(self._wf, trials)

        self._bilinearForms = [[_BilinearForm(self._wf, trials, trial, test, integrals) for test in self._tests] for trial in self._trials]
        self._linearForms = [_LinearForm(self._wf_b, test) for test in self._tests]
        self._offsets =mfem.intArray([0]+[t.mfem.space.GetTrueVSize() for t in self._trials])
        self._offsets.PartialSum()

    def __divideTerm(self, weakform, trials):
        d = {}
        for trial in trials:
            d[trial] = 0
            for gt in grad(trial):
                d[gt] = 0
        c = weakform.subs(d)
        return sp.simplify(weakform-c), -c

    def update(self, x_gfs, dt, coeffs_t):
        self._coeffs.update(coeffs_t)
        self.__updateResidual(x_gfs, dt)
        self.__updateJacobian(dt)
        return self._K, self._b, self._gK
        #return self._K.CreateMonolithic(), self._b, self._gK.CreateMonolithic()

    def __updateResidual(self, x, dt):
        """Calculate M, K, b"""
        self._bv = [lf.getDofs(self._coeffs) for lf in self._linearForms]
        self._k = [[b.getMatrix(dt, self._coeffs, x[i], self._bv[j]) for j, b in enumerate(bs)] for i, bs in enumerate(self._bilinearForms)]

        self._K = mfem.BlockOperator(self._offsets)
        for i, trial in enumerate(self._trials):
            for j, test in enumerate(self._tests):
                self._K.SetBlock(i, j, self._k[j][i])

        self._b = mfem.BlockVector(self._offsets)
        for j, b in enumerate(self._bv):
            b.GetTrueDofs(self._b.GetBlock(j))

    def __updateJacobian(self, dt):
        self._gk = [[b.getMatrixJacobian(dt, self._coeffs) for b in bs] for bs in self._bilinearForms]

        self._gK = mfem.BlockOperator(self._offsets)
        for i, trial in enumerate(self._trials):
            for j, test in enumerate(self._tests):
                self._gK.SetBlock(i, j, self._gk[j][i])

class _LinearForm:
    def __init__(self, wf, test):
        self._test = test

        self._parser = []

        self._scoef_V1 = self.__sympyCoeff1D(wf, test, test_deriv=True)
        self._parser.append(_coefParser(self._scoef_V1, dV))

        self._scoef_V2 = self.__sympyCoeff1D(wf, test)
        self._parser.append(_coefParser(self._scoef_V2, dV))

        self._scoef_S = self.__sympyCoeff1D(wf, test)
        self._parser.append(_coefParser(self._scoef_S, dS))

    def getDofs(self, coeffs):
        self._integ_V, self._integ_S = [], []

        # f.dot(∇v) * dV term
        if self._parser[0].valid:
            self._coef_V1 = self._parser[0].eval(coeffs)
            self._integ_V.append(mfem.DomainLFGradIntegrator(self._coef_V1))

        # f * v * dV term
        if self._parser[1].valid:
            self._coef_V2 = self._parser[1].eval(coeffs)
            self._integ_V.append(mfem.DomainLFIntegrator(self._coef_V2))

        # f * v * dS term
        if self._parser[2].valid:
            self._coef_S = self._parser[2].eval(coeffs)
            self._integ_S.append(mfem.BoundaryLFIntegrator(self._coef_S))

        return self.__linearForm(self._test.mfem.space, domainInteg=self._integ_V, boundaryInteg=self._integ_S)

    @classmethod
    def __sympyCoeff1D(cls, wf, test, test_deriv=False):
        if test_deriv is False:
            return wf.diff(test)
        else:
            return [wf.diff(t) for t in grad(test)]

    @staticmethod
    def __linearForm(space, domainInteg, boundaryInteg):
        # create Linear form of mfem
        b = mfem.LinearForm(space)
        for i in domainInteg:
            b.AddDomainIntegrator(i)
        for i in boundaryInteg:
            b.AddBoundaryIntegrator(i)
        b.Assemble()
        return b

class _BilinearForm:
    def __init__(self, wf, trials, trial, test, integrals):
        self._weakform = wf
        self._trial = trial
        self._test = test

        stiff_coef = self._getCoeffs(wf, trials, trial, test)
        self._stiff = _BilinearForm_time(trial, test, stiff_coef, integrals)

        coef_jac = self._getJacobiCoeffs(self._weakform, trial, test)
        coef_jac = self.__diff(coef_jac, stiff_coef)
        self._jac = _BilinearForm_time(trial, test, coef_jac, integrals)

        self.isNonlinear = sum(["trial_" in str(c) for c in stiff_coef]) != 0

    def __diff(self, jac, stiff):
        if isinstance(jac, (list, tuple)):
            return [self.__diff(j, s) for j,s in zip(jac, stiff)]
        else:
            return jac-stiff

    def getMatrix(self, dt, coeffs, x=None, b=None):
        self._stiff_mat = self._stiff.getMatrix(dt, coeffs, x, b)
        return self._stiff_mat
    
    def getMatrixJacobian(self, dt, coeffs):
        return self._jac.getMatrix(dt, coeffs) + self._stiff_mat

    @classmethod
    def _getCoeffs(cls, wf, trials, trial, test):
        vars, grads = [], []
        for tri in trials:
            vars.append(tri)
        for tri in trials:
            for gt in grad(tri):
                grads.append(gt)

        c0 = cls.__getCoeffs_single(wf.diff(test), vars, grads, trial)
        v0 = [cls.__getCoeffs_single(wf.diff(test), vars, grads, gt) for gt in grad(trial)]
        c1 = [cls.__getCoeffs_single(wf.diff(gt), vars, grads, trial) for gt in grad(test)]
        v1 = [[cls.__getCoeffs_single(wf.diff(gtest), vars, grads, gtrial) for gtrial in grad(trial)] for gtest in grad(test)]
        return c0, v0, c1, v1 # coef for u*v, ∇u*v, u*∇v, ∇u*∇v

    @classmethod
    def __getCoeffs_single(cls, wf, vars, grads, var):
        p = sp.poly(wf, *vars, *grads)
        index = (vars+grads).index(var)

        c = 0
        for args in p.as_dict().keys():
            if args[index] != 0:
                args_grads = args[len(vars):]
                # It is preffered to use ∇u term as trial function.
                if index >= len(vars):
                    value = p.nth(*args) * args[index]/sum(args_grads)
                elif sum(args_grads) == 0:
                    value = p.nth(*args) * args[index]/sum(args)
                else:
                    value = 0
                for order, v in zip(args, vars+grads):
                    if v == var:
                        order -= 1
                    value *=  v**order
                c += value
        return c

    @classmethod
    def _getJacobiCoeffs(cls, wf, trial, test):
        diffs_trial = grad(trial)
        diffs_test = grad(test)
        c1 = wf.diff(trial).diff(test)
        c2 = [wf.diff(t).diff(test) for t in diffs_trial]
        c3 = [wf.diff(trial).diff(t) for t in diffs_test]
        c4 = [[wf.diff(t).diff(t2) for t in diffs_trial] for t2 in diffs_test]
        return c1, c2, c3, c4

class _BilinearForm_time:
    def __init__(self, trial, test, coef, integrals, divide=True):
        self._divide=divide
        if divide:
            coef_tt, coef_t, coef_c = self.__timeDerivativeTerm(coef)
            self._coef_t = _BilinearFormMatrix(trial, test, coef_t, integrals)
            self._coef_c = _BilinearFormMatrix(trial, test, coef_c, integrals)
        else:
            self._coef = _BilinearFormMatrix(trial, test, coef, integrals)

    def __timeDerivativeTerm(self, weakforms, order=None):
        if order is None:
            return [self.__timeDerivativeTerm(weakforms, i) for i in [2,1,0]]
        if isinstance(weakforms, (tuple, list)):
            return [self.__timeDerivativeTerm(w, order) for w in weakforms]
        return sp.Poly(weakforms, 1/dt).nth(order)

    def getMatrix(self, dt, coeffs, x=None, b=None):
        if self._divide:
            self._mat_t = self._coef_t.getMatrix(coeffs, x, b)
            self._mat_c = self._coef_c.getMatrix(coeffs, x, b)
            return self._mat_t*(1/dt) + self._mat_c
        else:
            return self._coef.getMatrix(coeffs, x, b)

class _BilinearFormMatrix:
    def __init__(self, trial, test, coeffs, integrals):
        self._trial = trial
        self._test = test

        self._nonlinear = self.__isNonlinear(coeffs)
        self._mat = None

        self._parsers = []
        for c in coeffs:
            self._parsers.append(_coefParser(c, dV))

        self._parsers_S = []
        for c in coeffs:
            self._parsers_S.append(_coefParser(c, dS))

    def __isNonlinear(self, coeffs):
        for c in coeffs:
            if "trial_" in str(c):
                return True
            if "prev_" in str(c):
                return True
        return False
        
    def getMatrix(self, coeffs, x=None, b=None):
        if not self._nonlinear and self._mat is not None:
            return self._mat
        self._integ = []
        if self._parsers[0].valid:
            self._coef_V0 = self._parsers[0].eval(coeffs)
            self._integ.append(mfem.MixedScalarMassIntegrator(self._coef_V0))
        if self._parsers[1].valid:
            self._coef_V1 = self._parsers[1].eval(coeffs)
            self._integ.append(mfem.MixedDirectionalDerivativeIntegrator(self._coef_V1))
        if self._parsers[2].valid:
            self._coef_V2 = self._parsers[2].eval(coeffs)
            self._coef_V2_ =mfem.ScalarVectorProductCoefficient(-1.0, self._coef_V2)
            self._integ.append(mfem.MixedScalarWeakDivergenceIntegrator(self._coef_V2_))
        if self._parsers[3].valid:
            self._coef_V3 = self._parsers[3].eval(coeffs)
            self._integ.append(mfem.MixedGradGradIntegrator(self._coef_V3))

        self._integ_S = []
        if self._parsers_S[0].valid:
            self._coef_S0 = self._parsers_S[0].eval(coeffs)
            self._integ_S.append(mfem.MixedScalarMassIntegrator(self._coef_S0))
        if self._parsers_S[1].valid:
            self._coef_S1 = self._parsers_S[1].eval(coeffs)
            self._integ_S.append(mfem.MixedDirectionalDerivativeIntegrator(self._coef_S1))
        if self._parsers_S[2].valid:
            self._coef_S2 = self._parsers_S[2].eval(coeffs)
            self._coef_S2_ =mfem.ScalarVectorProductCoefficient(-1.0, self._coef_S2)
            self._integ_S.append(mfem.MixedScalarWeakDivergenceIntegrator(self._coef_S2_))
        if self._parsers_S[3].valid:
            self._coef_S3 = self._parsers_S[3].eval(coeffs)
            self._integ_S.append(mfem.MixedGradGradIntegrator(self._coef_S3))

        self._mat = self.__generateMatrix(x,b,domainInteg=self._integ, boundaryInteg=self._integ_S)
        return self._mat

    def __generateMatrix(self, x, b, domainInteg, boundaryInteg):
        if self._trial.mfem.space == self._test.mfem.space:
            return self._bilinearForm(self._trial.mfem.space, self._trial.mfem.ess_bdrs, x, b,
                                      domainInteg=domainInteg, boundaryInteg=boundaryInteg)
        else:
            return self._mixedBilinearForm(self._trial.mfem.space, self._test.mfem.space,
                                           self._trial.mfem.ess_bdrs, self._test.mfem.ess_bdrs, x, b, 
                                           domainInteg=domainInteg, boundaryInteg=boundaryInteg)

    @staticmethod
    def _bilinearForm(space, ess_bdrs=None, x=None, b=None, domainInteg=None, boundaryInteg=None):
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
        result = mfem.SparseMatrix(m.SpMat())
        result._bilin = m
        return result

    @staticmethod
    def _mixedBilinearForm(space1, space2, ess_bdrs1=None, ess_bdrs2=None, x=None, b=None, domainInteg=None, boundaryInteg=None):
        # create Bilinear form of mfem
        m = mfem.MixedBilinearForm(space1, space2)
        for i in domainInteg:
            m.AddDomainIntegrator(i)
        for i in boundaryInteg:
            mk = mfem.intArray()
            space2.ListToMarker(mfem.intArray([1]), 1, mk)
            print("marker", [mn for mn in mk])
            m.AddBoundaryIntegrator(i, mfem.intArray([1]))
        m.Assemble()
    
        if b is None:
            x = mfem.Vector(space1.GetVSize())
            b = mfem.Vector(space2.GetVSize())
        m.EliminateTrialDofs(ess_bdrs1, x, b)
        m.EliminateTestDofs(ess_bdrs2)

        # set it to matrix
        m.Finalize()
        result = mfem.SparseMatrix(m.SpMat())
        result._bilin = m

        return result


class _coefParser:
    def __init__(self, scoef, domain=None):
        if domain is not None:
            scoef = self.__coeff(scoef, domain)
        self._empty = self.__is_zero(scoef)
        if self._empty:
            return

        if isinstance(scoef, (list, tuple)):
            if isinstance(scoef[0], (list, tuple)):
                self._type = "matrix"
                self._scalars = [[_coefParser(s) for s in sc] for sc in scoef]
            else:
                self._type = "vector"
                self._scalars = [_coefParser(sc) for sc in scoef]
        else:
            self._type = "scalar"
            if not isinstance(scoef, (sp.Basic, sp.Matrix)):
                self._func = mfem.generateCoefficient(scoef)
                self._const = True
            else:
                scoef = self.__replaceFuncs(scoef)
                self._args = tuple(scoef.free_symbols)
                self._func = sp.lambdify(self._args, scoef)
                self._const = False

    def eval(self, coefs):
        if self._type == "vector":
            return mfem.VectorArrayCoefficient([s.eval(coefs) for s in self._scalars])
        if self._type == "matrix":
            return mfem.MatrixArrayCoefficient([[s.eval(coefs) for s in ss] for ss in self._scalars])
        if self._empty:
            return 0
        if self._const:
            return self._func
        else:
            res = self._func(*[coefs[str(a)] for a in self._args])
            if isinstance(res, (int, float, sp.Integer, sp.Float)):
                res = mfem.generateCoefficient(res)
            return res

    def __coeff(self, scoef, domain):
        if hasattr(scoef, "__iter__"):
            return [self.__coeff(s, domain) for s in scoef]
        else:
            return scoef.coeff(domain)

    def __is_zero(self, val):
        if isinstance(val, (tuple, list)):
            return np.array([self.__is_zero(v) for v in val]).all()
        else:
            return val==0

    def __replaceFuncs(self, wf):
        """Used from sympyCoeff"""
        if not isinstance(wf, (sp.Basic, sp.Matrix)):
            return wf
        for f in wf.atoms(sp.Function):
            for var in [x,y,z]:
                wf = wf.subs(f.diff(var), sp.Symbol(f.func.name+"_"+str(var)))
            wf = wf.subs(f, sp.Symbol(f.func.name))
        return wf

    @property
    def valid(self):
        return not self._empty