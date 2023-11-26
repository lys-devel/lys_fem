from . import mfem, util
import sympy as sp

x, y, z, t, dV, dS = sp.symbols("x,y,z,t,dV,dS")

def grad(v):
    if z in v.free_symbols:
        return sp.Matrix([[v.diff("x")], [v.diff("y")], [v.diff("z")]])
    elif y in v.free_symbols:
        return sp.Matrix([[v.diff("x")], [v.diff("y")]])
    else:
        return sp.Matrix([[v.diff("x")]])

def TrialFunction(name, mesh, ess_bdrs, value, order=1):
    result = sp.Function("trial_"+name)
    if mesh.SpaceDimension() == 1:
        result = result(x,t)
    if mesh.SpaceDimension() == 2:
        result = result(x,t,t)
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
    def x(self):
        return self._x

def TestFunction(trial):
    result = sp.Function("test_"+trial.func.name.replace("trial_", ""))(*trial.free_symbols)
    if hasattr(trial, "mfem"):
        result.mfem = trial.mfem
    return result

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


class WeakformParser:
    def __init__(self, wf, trials, coeffs):
        self._wf = wf
        self._trials = trials
        self._tests = [TestFunction(t) for t in trials]
        self._coeffs = coeffs
        self._offsets =mfem.intArray([0]+[t.mfem.space.GetTrueVSize() for t in self._trials])
        self._offsets.PartialSum()

    def update(self):
        self._M = mfem.BlockMatrix(self._offsets)
        self._K = mfem.BlockMatrix(self._offsets)
        self._x = mfem.BlockVector(self._offsets)
        self._b = mfem.BlockVector(self._offsets)
        for i, trial in enumerate(self._trials):
            self._x.GetBlock(i).Set(1, trial.mfem.x)
            for j, test in enumerate(self._tests):
                self._M.SetBlock(i,j,getMatrix(self._wf, trial, test, self._coeffs, deriv_t=1))
                K = getMatrix(self._wf, trial, test, self._coeffs, True, True)
                self._K.SetBlock(i,j,K)
            if i==j:
                self._b.GetBlock(i).Set(1, getVector(self._wf, test, self._coeffs, K, trial.mfem.x))
        return mfem.SparseMatrix(self._M.CreateMonolithic()), mfem.SparseMatrix(self._K.CreateMonolithic()), self._x, self._b


def getVector(wf, test, coeffs, K, x0):
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
    return util.linearForm(test.mfem.space, test.mfem.ess_tdof, K, x0, domainInteg=[integ_V2], boundaryInteg=integ_S)


def getMatrix(wf, trial, test, coeffs, trial_deriv=False, test_deriv=False, deriv_t=0):
    trial_t=trial
    for _ in range(deriv_t):
        trial_t=trial_t.diff(t)
    scoef_V = sympyCoeff(wf.coeff(dV), trial_t, test, trial_deriv, test_deriv)
    coef_V = SubsCoeff(scoef_V, coeffs)
    integ_V = _bilinearIntegrator(coef_V, trial_deriv, test_deriv)
    integ_S = None
    if trial.mfem.space == test.mfem.space:
        return util.bilinearForm(trial.mfem.space, trial.mfem.ess_tdof, domainInteg=integ_V,boundaryInteg=integ_S)
    else:
        return util.mixedBilinearForm(trial.mfem.space, test.mfem.space, trial.mfem.ess_tdof, test.mfem.ess_tdof, integ_domain=integ_V, boundaryInteg=integ_S)


def _bilinearIntegrator(coef, trial_deriv=False, test_deriv=False):
    if trial_deriv:
        if test_deriv:
            return mfem.MixedGradGradIntegrator(coef)
        else:
            return mfem.MixedDirectionalDerivativeIntegrator(coef)
    else:
        if test_deriv:
            return mfem.MixedScalarWeakDivergenceIntegrator(coef)
        else:
            return mfem.MixedScalarMassIntegrator(coef)

def _linearIntegrator(coef, test_deriv=False, boundary=False):
    if boundary:
        return mfem.BoundaryLFIntegrator(coef)
    if test_deriv:
        return mfem.DomainLFGradIntegrator(coef)
    else:
        return mfem.DomainLFIntegrator(coef)

def SubsCoeff(scoef, coefs):
    """
    Substitute sympy scoef object by coefs dictionary.
    """
    args = tuple(scoef.free_symbols)
    scoef = sp.lambdify(args, scoef)(*[coefs[str(a)] for a in args])
    if isinstance(scoef, sp.Matrix):
        shape = scoef.shape
        if len(shape) == 2:
            res = mfem.MatrixArrayCoefficient(shape[0])
            for i in range(shape[0]):
                for j in range(shape[1]): 
                    res.Set(i,j,scoef[i,j])
        elif len(shape) == 1:
            res = mfem.VectorArrayCoefficient(shape[0])
            for i in range(shape[0]):
                res.Set(i, scoef[i])
    else:
        res = scoef
        if res == 0:
            res = mfem.ConstantCoefficient(0)
    return res


def replaceFuncs(wf, removeTrial=False):
    for f in wf.atoms(sp.Function):
        if removeTrial and "trial_" in f.func.name:
            enable = 0
        else:
            enable = 1
        for var in [x,y,z,t]:
            wf = wf.subs(f.diff(var), sp.Symbol(f.func.name+"_"+str(var))*enable)
        wf = wf.subs(f, sp.Symbol(f.func.name)*enable)
    return wf


def sympyCoeff1D(wf, test, test_deriv=False):
    wf = replaceFuncs(wf, True)
    if test_deriv is False:
        return wf.coeff(sp.Symbol(test.func.name))
    else:
        return sp.Matrix([wf.coeff(replaceFuncs(t)) for t in grad(test)])


def sympyCoeff(wf, trial, test, trial_deriv=False, test_deriv=False):
    wf = replaceFuncs(wf)
    if trial_deriv is False:
        wf_t = _coeff(wf, replaceFuncs(trial))
        if test_deriv is False:
            return wf_t.coeff(sp.Symbol(test.func.name))
        else:
            return sp.Matrix([wf_t.coeff(replaceFuncs(t)) for t in grad(test)])
    else:
        res = []
        for tri in grad(trial):
            wf_t = _coeff(wf, replaceFuncs(tri))
            if test_deriv is False:
                res.append(wf_t.coeff(replaceFuncs(test)))
            else:
                c = [wf_t.coeff(replaceFuncs(t)) for t in grad(test)]
                if len(c) == 1:
                    res.append(c[0])
                else:
                    res.append(c)
        if len(res) == 1:
            return res[0]
        else:
            return sp.Matrix(res)    

def _coeff(expr, x):
    if expr == 0:
        return sp.core.numbers.Zero()
    p = sp.poly(expr, x)
    ac = p.all_coeffs()
    res = sp.core.numbers.Zero()
    for order in range(p.degree()):
        res += ac[order] * x**(p.degree() - order - 1)
    return res