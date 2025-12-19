import numpy as np

from lys_fem import FEMModel, DomainCondition, Coef, GeometrySelection, util, time
from lys_fem.util import grad, dx
from . import InitialCondition, DirichletBoundary


class ThermoelasticStress(DomainCondition):
    className = "ThermoelasticStress"

    def __init__(self, T="T", alpha="alpha", geometries="all", **kwargs):
        super().__init__(geometries = geometries, **kwargs)
        self["T"] = Coef(T, description="Temperature (K)")
        self["alpha"] = Coef(alpha, shape=(3,3), description="Thermal expansion coef. (1/K)")


class DeformationPotential(DomainCondition):
    className = "DeformationPotential"

    def __init__(self, n_e="n_e", n_h="n_h", geometries="all", *args, **kwargs):
        super().__init__(geometries=geometries, *args, **kwargs)
        self["n_e"] = Coef(n_e, description="Electron carrier density (1/m3)")
        self["n_h"] = Coef(n_h, description="Hole carrier density (1/m3)")


class InversePiezoelectricity(DomainCondition):
    className = "InversePiezoelectricity"

    def __init__(self, E="E", geometries="all", *args, **kwargs):
        super().__init__(geometries=geometries, *args, **kwargs)
        self["E"] = Coef(E, (3,), description="Electric field (V/m)")


class PerfectlyMatchedLayer(DomainCondition):
    className = "PML_elasticity"

    def __init__(self, sigma="sigma_PML", geometries="all", *args, **kwargs):
        super().__init__(geometries=geometries, *args, **kwargs)
        self["sigma"] = Coef(sigma, (3,), description="sigma for PML")


class ElasticModel(FEMModel):
    className = "Elasticity"
    boundaryConditionTypes = [DirichletBoundary]
    domainConditionTypes = [ThermoelasticStress, DeformationPotential, InversePiezoelectricity, PerfectlyMatchedLayer]
    initialConditionTypes = [InitialCondition]

    def __init__(self, nvar=3, discretization="NewmarkBeta", C="C", rho="rho", *args, **kwargs):
        super().__init__(nvar, *args, varName="u", discretization=discretization, **kwargs)
        self["rho"] = Coef(rho, description="Density (kg/m3)")
        self["C"] = Coef(C, shape=(3,3,3,3), description="Elastic Constant (Pa)")
    
    def functionSpaces(self):
        # geometry for default equation
        res = super().functionSpaces()

        # define PML spaces
        kwargs = {"fetype": "H1", "order": 1, "isScalar": False, "size": self._varDim}
        for pml in self.domainConditions.get(PerfectlyMatchedLayer):
            for d in range(self.fem.dimension):
                res.append(util.FunctionSpace(self._varName+"_w"+str(d), geometries=pml.geometries, **kwargs))
            res.append(util.FunctionSpace(self._varName+"_U", geometries=pml.geometries, **kwargs))
        return res

    def initialValues(self, params):
        res = super().initialValues(params)
        for pml in self.domainConditions.get(PerfectlyMatchedLayer):
            for d in range(self.fem.dimension):
                res.append(util.NGSFunction([0]*self.variableDimension))
            res.append(util.NGSFunction([0]*self.variableDimension))
        return res

    def _initialVelocities(self, params):
        res = super().initialVelocities(params)
        for pml in self.domainConditions.get(PerfectlyMatchedLayer):
            for d in range(self.fem.dimension):
                res.append(util.NGSFunction([0]*self.variableDimension))
            res.append(util.NGSFunction([0]*self.variableDimension))
        return res

    def discretizes(self, dti):
        d = super().discretize(dti)
        for v in self.functionSpaces():
            if v.name != self._varName and (self._varName+"_U" in v.name or self._varName+"_w" in v.name):
                trial = util.trial(v)
                d.update(time.BackwardEuler.generateWeakforms(trial, dti))
        return d
    
    def weakform(self, vars, mat):
        C, rho = mat[self.C], mat[self.rho]
        wf = 0

        u,v = vars[self.variableName]
        gu, gv = grad(u), grad(v)
        
        wf += rho * u.tt.dot(v) * dx
        wf += gu.ddot(C.ddot(gv)) * dx

        for te in self.domainConditions.get(ThermoelasticStress):
            alpha, T = mat[te.alpha], mat[te.T]
            wf += T*C.ddot(alpha).ddot(gv)*dx(te.geometries)

        for df in self.domainConditions.get(DeformationPotential):
            d_n, d_p = mat["-d_e*e"], mat["-d_h*e"]
            n,p = mat[df.n_e], mat[df.n_h]
            I = mat[np.eye(3)]
            wf += (d_n*n - d_p*p)*gv.ddot(I)*dx(df.geometries)
        
        for pe in self.domainConditions.get(InversePiezoelectricity):
            e, E = mat["e_piezo"], mat[pe.E]
            wf -= -E.dot(e).ddot(gv)*dx(pe.geometries)

        for pml in self.domainConditions.get(PerfectlyMatchedLayer):
            sigma = mat[pml.sigma]
            a = sigma.dot(util.eval([1,1,1]))
            b = sigma[0] * sigma[1] + sigma[1] * sigma[2] + sigma[2] * sigma[0]
            c = sigma[0] * sigma[1] * sigma[2]

            C1 = a*C - util.einsum("j,ijkl->ijkl", sigma, C) - util.einsum("l,ijkl->ijkl", sigma, C)
            A = np.zeros((3,3,3))
            A[0,1,2] = A[0,2,1] = A[1,2,0] = A[1,0,2] = A[2,0,1] = A[2,1,0] = 0.5
            cb = util.einsum("ijk,j,k->i", util.eval(A), sigma, sigma)
            C2 = util.einsum("l,ijkl->ijkl", cb, C)

            U, U_test = vars[self._varName+"_U"]
            w = util.NGSFunction([self._tnt(vars, i, 0) for i in range(3)], name="w")
            wt = util.NGSFunction([self._tnt(vars, i, 0, True) for i in range(3)], name="w")
            w_test = util.NGSFunction([self._tnt(vars, i, 1) for i in range(3)], name="w")

            wf += (U.t - u).dot(U_test)*dx(pml.geometries)
            wf += (wt + util.einsum("j,ij->ij", sigma, w) - C1.ddot(grad(u)) - C2.ddot(grad(U))).ddot(w_test)*dx(pml.geometries)
            wf += rho * (a*u.t + b*u + c*U).dot(v)*dx(pml.geometries) + w.ddot(grad(v))*dx(pml.geometries)

        return wf
    
    def _tnt(self, vars, d, index, deriv=False):
        if d < self.fem.dimension:
            res= vars[self._varName+"_w"+str(d)][index]
            if deriv:
                return res.t
            else:
                return res
        else:
            return util.NGSFunction((0,0,0), name="zero")