import numpy as np

from lys_fem import FEMModel, DomainCondition, Coef, GeometrySelection, util
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
        geometries = list(self._geometries)
        for pml in self.domainConditions.get(PerfectlyMatchedLayer):
            geometries = [g for g in geometries if g not in pml.geometries]
        geometries = GeometrySelection(selection=geometries, parent=self)

        # default u
        res = [self.equation.functionSpaces(self.boundaryConditions.dirichlet, geometries=geometries)]

        # define PML spaces
        kwargs = {"fetype": "H1", "order": 1, "isScalar": False, "size": self._varDim}
        for pml in self.domainConditions.get(PerfectlyMatchedLayer):
            for d in range(self.fem.dimension):
                res.append(util.FunctionSpace(self._varName+"_u"+str(d), geometries=pml.geometries, **kwargs))
                res.append(util.FunctionSpace(self._varName+"_p"+str(d), geometries=pml.geometries, **kwargs))
        return res

    def initialValues(self, params):
        res = super().initialValues(params)
        for pml in self.domainConditions.get(PerfectlyMatchedLayer):
            for d in range(self.fem.dimension):
                res.append(res[0]/self.variableDimension)
                res.append(util.NGSFunction([0]*self.variableDimension))
        return res

    def _initialVelocities(self, params):
        res = super().initialVelocities(params)
        for pml in self.domainConditions.get(PerfectlyMatchedLayer):
            for d in range(self.fem.dimension):
                res.append(res[0]/self.variableDimension)
                res.append(util.NGSFunction([0]*self.variableDimension))
        return res
    
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
            q = util.NGSFunction([self._tnt(vars, "u", i, 0) for i in range(3)], name="q")
            qt = util.NGSFunction([self._tnt(vars, "u", i, 0, 1) for i in range(3)], name="qt")
            qtt = util.NGSFunction([self._tnt(vars, "u", i, 0, 2) for i in range(3)], name="qtt")
            q_test = util.NGSFunction([self._tnt(vars, "u", i, 1) for i in range(3)], name="q_test")
            p = util.NGSFunction([self._tnt(vars, "p", i, 0) for i in range(3)], name="p")
            p_test = util.NGSFunction([self._tnt(vars, "p", i, 1) for i in range(3)], name="p_test")

            u = q.dot(util.eval([1,1,1]))
            I = util.eval(np.eye(3), name="I")

            stress = C.ddot(grad(u) + p)
            wf += (rho*(qtt + sigma*qt).ddot(q_test) - stress.ddot(I.ddot(grad(q_test))))*dx(pml.geometries)

        return wf
    
    def _tnt(self, vars, var, d, index, deriv=None):
        if d < self.fem.dimension:
            res = vars[self._varName+"_"+var+str(d)][index]
            if deriv == 1:
                return res.t
            elif deriv == 2:
                return res.tt
            else:
                return res
        else:
            return util.NGSFunction((0,0,0), name="zero")