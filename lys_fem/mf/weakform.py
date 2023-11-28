from . import mfem, util
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
    result = sp.Function("trial_"+name)
    if mesh.SpaceDimension() == 1:
        result = result(x,t)
    if mesh.SpaceDimension() == 2:
        result = result(x,y,t)
    if mesh.SpaceDimension() == 3:
        result = result(x,y,z,t)
    result.coeff = sp.Symbol("coeff_"+name)
    if ess_bdrs is not None:
        result.mfem = _MFEMInfo(name, mesh, ess_bdrs, value, order)
    return result

class _MFEMInfo:
    def __init__(self, name, mesh, ess_bdrs, value, order=1):
        self._name = name
        fec = mfem.H1_FECollection(order, mesh.Dimension())
        self._space = mfem.FiniteElementSpace(mesh, fec, 1)
        self._ess_tdof = self.__getEssentialTrueDofs(ess_bdrs)
        self.setValue(value)

    def __getEssentialTrueDofs(self, ess_bdrs):
        ess_bdr = mfem.intArray(self.space.GetMesh().bdr_attributes.Max())
        ess_bdr.Assign(0)
        for i in ess_bdrs:
            ess_bdr[i - 1] = 1
        self._ess_bdrs = ess_bdr
        ess_tdof_list = mfem.intArray()
        self.space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list, 1)
        return ess_tdof_list
    
    def setValue(self, coef):
        self._x = mfem.Vector()
        x_gf = mfem.GridFunction(self._space)
        x_gf.ProjectCoefficient(coef)
        x_gf.GetTrueDofs(self._x)

    @property
    def space(self):
        return self._space
    
    @property
    def ess_tdof(self):
        return self._ess_tdof
    
    @property
    def ess_bdrs(self):
        return self._ess_bdrs

    @property
    def x(self):
        return self._x

def TestFunction(trial):
    if isinstance(trial, sp.Matrix):
        return sp.Matrix([TestFunction(t) for t in trial])
    result = sp.Function("test_"+trial.func.name.replace("trial_", ""))(*trial.free_symbols)
    if hasattr(trial, "mfem"):
        result.mfem = trial.mfem
    return result


class WeakformParser:
    def __init__(self, wf, trials, coeffs):
        self._wf = wf
        self._trials = trials
        self._tests = [TestFunction(t) for t in trials]
        self._coeffs = coeffs
        self._offsets =mfem.intArray([0]+[t.mfem.space.GetTrueVSize() for t in self._trials])
        self._offsets.PartialSum()

    def update(self):
        self._x = mfem.BlockVector(self._offsets)
        self._b = mfem.BlockVector(self._offsets)
        for i, trial in enumerate(self._trials):
            self._x.GetBlock(i).Set(1, trial.mfem.x)
        for j, test in enumerate(self._tests):
            self._b.GetBlock(j).Set(1, getVector(self._wf, test, self._coeffs))

        self._m = [[getMatrix(self._wf, trial, test, self._coeffs, deriv_t=1) for test in self._tests] for trial in self._trials]
        self._k = [[getMatrix(self._wf, trial, test, self._coeffs, self._x.GetBlock(i), self._b.GetBlock(j), True, True) for j, test in enumerate(self._tests)] for i, trial in enumerate(self._trials)]

        self._M = mfem.BlockOperator(self._offsets)
        self._K = mfem.BlockOperator(self._offsets)
        for i, trial in enumerate(self._trials):
            for j, test in enumerate(self._tests):
                self._M.SetBlock(i, j, self._m[j][i])
                self._K.SetBlock(i, j, self._k[j][i])

        return self._M, self._K, self._x, self._b
        #return mfem.SparseMatrix(self._M.CreateMonolithic()), mfem.SparseMatrix(self._K.CreateMonolithic()), self._x, self._b


def getVector(wf, test, coeffs):#, K, x0):
    # f * ∇v * dV term
    #scoef_V1 = sympyCoeff1D(wf.coeff(dV), test, True)
    #coef_V1 = SubsCoeff(scoef_V1, coeffs)
    #integ_V1 = _linearIntegrator(coef_V1, True)

    # f * v * dV term
    scoef_V2 = sympyCoeff1D(wf.coeff(dV), test, test_deriv=False)
    coef_V2 = SubsCoeff(scoef_V2, coeffs)
    integ_V2 = _linearIntegrator(coef_V2, False)

    # f * v * dS term
    scoef_S = sympyCoeff1D(wf.coeff(dS), test)
    coef_S = SubsCoeff(scoef_S, coeffs)
    integ_S = _linearIntegrator(coef_S, boundary=True)
    return util.linearForm(test.mfem.space, domainInteg=[integ_V2], boundaryInteg=integ_S)

def _linearIntegrator(coef, test_deriv=False, boundary=False):
    if boundary:
        return mfem.BoundaryLFIntegrator(coef)
    if test_deriv:
        return mfem.DomainLFGradIntegrator(coef)
    else:
        return mfem.DomainLFIntegrator(coef)

def getMatrix(wf, trial, test, coeffs, x=None, b=None, trial_deriv=False, test_deriv=False, deriv_t=0):
    trial_t=trial
    for _ in range(deriv_t):
        trial_t=trial_t.diff(t)
    scoef_V = sympyCoeff(wf.coeff(dV), trial_t, test, trial_deriv, test_deriv)
    print("sympy coefficient for weakform", scoef_V)
    coef_V = SubsCoeff(scoef_V, coeffs)
    integ_V = _bilinearIntegrator(coef_V, trial_deriv, test_deriv)
    integ_S = None
    if trial.mfem.space == test.mfem.space:
        return util.bilinearForm(trial.mfem.space, trial.mfem.ess_bdrs, x, b, domainInteg=integ_V,boundaryInteg=integ_S)
    else:
        return util.mixedBilinearForm(trial.mfem.space, test.mfem.space, trial.mfem.ess_bdrs, test.mfem.ess_bdrs, x, b, domainInteg=integ_V, boundaryInteg=integ_S)


def _bilinearIntegrator(coef, trial_deriv=False, test_deriv=False):
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
        

def SubsCoeff(scoef, coefs):
    """
    Substitute sympy coef object by coefs dictionary.
    """
    if isinstance(scoef, sp.Matrix):
        shape = scoef.shape
        if len(shape) == 2:
            res = mfem.MatrixArrayCoefficient(shape[0])
            for i in range(shape[0]):
                for j in range(shape[1]):
                    res.Set(i,j,SubsCoeff(scoef[i,j], coefs))
        elif len(shape) == 1:
            res = mfem.VectorArrayCoefficient(shape[0])
            for i in range(shape[0]):
                res.Set(i, SubsCoeff(scoef[i], coefs))
    else:
        args = tuple(scoef.free_symbols)
        res = sp.lambdify(args, scoef)(*[coefs[str(a)] for a in args])
        if res == 0:
            res = mfem.ConstantCoefficient(0)
    return res


def sympyCoeff1D(wf, test, test_deriv=False):
    wf = _replaceFuncs(wf, True)
    if test_deriv is False:
        return wf.coeff(sp.Symbol(test.func.name))
    else:
        return sp.Matrix([wf.coeff(_replaceFuncs(t)) for t in grad(test)])


def sympyCoeff(wf, trial, test, trial_deriv=False, test_deriv=False):
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
        wf_t = _coeff(wf, _replaceFuncs(trial))
        if test_deriv is False:
            return wf_t.coeff(sp.Symbol(test.func.name))
        else:
            return sp.Matrix([wf_t.coeff(_replaceFuncs(t)) for t in grad(test)])
    else:
        res = []
        for tri in grad(trial):
            wf_t = _coeff(wf, _replaceFuncs(tri))
            if test_deriv is False:
                res.append(wf_t.coeff(_replaceFuncs(test)))
            else:
                c = [wf_t.coeff(_replaceFuncs(t)) for t in grad(test)]
                if len(c) == 1:
                    res.append(c[0])
                else:
                    res.append(c)
        if len(res) == 1:
            return res[0]
        else:
            return sp.Matrix(res)


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


def _coeff(expr, x):
    """Used from sympyCoeff"""
    if expr == 0:
        return sp.core.numbers.Zero()
    p = sp.poly(expr, x)
    ac = p.all_coeffs()
    res = sp.core.numbers.Zero()
    for order in range(p.degree()):
        res += ac[order] * x**(p.degree() - order - 1)
    return res


class Coefficient:
    def __init__(self, value=None):
        if value is None:
            self._value = mfem.ConstantCoefficient(1)
        else:  
            self._value = value

    @property
    def value(self):
        return self._value
    
    def __neg__(self):
        return Coefficient(mfem.ProductCoefficient(-1, self.value))
        
    def __mul__(self, other):
        if isinstance(other, Coefficient):
            return Coefficient(mfem.ProductCoefficient(self.value, other.value))
        elif isinstance(other, TrialFunction):
            return other * self
        else: # other is float
            return Coefficient(mfem.ProductCoefficient(other, self.value))
        
    def __add__(self, other):
        if isinstance(other, Coefficient):
            return Coefficient(mfem.SumCoefficient(1, self.value, 1, other.value))

    def __sub__(self, other):
        if isinstance(other, Coefficient):
            return Coefficient(mfem.SumCoefficient(1, self.value, -1, other.value))
