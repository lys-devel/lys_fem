

from lys_fem import FEMFixedModel, DomainCondition, Coef, util
from lys_fem.util import g_e, e, mu_B, mu_0, dti, k_B, Ve, grad, dx
from . import DirichletBoundary


class ExternalMagneticField(DomainCondition):
    className = "External Magnetic Field"

    @classmethod
    def default(cls, fem, model):
        return ExternalMagneticField([0,0,0])

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Magnetic Field (T)")


class UniaxialAnisotropy(DomainCondition):
    className = "UniaxialAnisotropy"


class CubicAnisotropy(DomainCondition):
    className = "CubicAnisotropy"


class CubicMagnetoStriction(DomainCondition):
    className = "CubicMagnetoStriction"

    @classmethod
    def default(cls, fem, model):
        return CubicMagnetoStriction("u")

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Displacement field (m)")


class CubicMagnetoRotationCoupling(DomainCondition):
    className = "CubicMagnetoRotationCoupling"

    @classmethod
    def default(cls, fem, model):
        return CubicMagnetoRotationCoupling("u")

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Displacement field (m)")


class BarnettEffect(DomainCondition):
    className = "BarnettEffect"

    @classmethod
    def default(cls, fem, model):
        return BarnettEffect("u")

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Displacement field (m)")


class MagneticScalarPotential(DomainCondition):
    className = "MagneticScalarPotential"
    def __init__(self, values, geometries="all", *args, **kwargs):
        values = Coef(values, description="Magnetic scalar potential (A)")
        super().__init__(values=values, geometries=geometries, *args, **kwargs)

    @classmethod
    def default(cls, fem, model):
        return MagneticScalarPotential(0)

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Magnetic scalar potential (A)")


class SpinTransferTorque(DomainCondition):
    className = "SpinTransferTorque"

    @classmethod
    def default(cls, fem, model):
        return cls([0]*fem.dimension)

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Spin polarized current (A/m^2)")


class ThermalFluctuation(DomainCondition):
    className = "ThermalFluctuation"

    @classmethod
    def default(cls, fem, model):
        return cls(T=0, R=[0,0,0])

    def widget(self, fem, canvas):
        from .widgets import ThermalFluctuationWidget
        return ThermalFluctuationWidget(self)


class LLGModel(FEMFixedModel):
    className = "LLG"
    domainConditionTypes = [ExternalMagneticField, UniaxialAnisotropy, CubicAnisotropy, MagneticScalarPotential, CubicMagnetoStriction, CubicMagnetoRotationCoupling, BarnettEffect, SpinTransferTorque, ThermalFluctuation]
    boundaryConditionTypes = [DirichletBoundary]

    def __init__(self, *args, order=2, **kwargs):
        super().__init__(3, *args, order=order, varName="m", valType="v", **kwargs)

    def functionSpaces(self):
        fes = super().functionSpaces()[0]

        kwargs = {"size": 1, "isScalar": True, "order": self.order-1, "fetype": self.fetype}
        return [fes, util.FunctionSpace(self.variableName+"_lam", geometries=self.geometries, **kwargs)]

    def initialValues(self, params):
        x0 = super().initialValues(params)
        x0.append(util.eval(0))
        return x0

    def initialVelocities(self, params):
        v0 = super().initialVelocities(params)
        v0.append(util.eval(0))
        return v0

    def discretize(self, dti):
        d = {}
        for v in self.functionSpaces():
            if "_lam" in v.name:
                continue
            trial = util.trial(v)
            mn = util.prev(trial)
            d[trial] = mn
            d[trial.t] = trial
        return d

    def updater(self, dti):
        d = super().updater(dti)
        for v in self.functionSpaces():
            if "_lam" in v.name:
                continue
            trial = util.trial(v)
            m = util.prev(trial) + trial/dti
            d[trial] = m/util.norm(m)
            d[trial.t] = trial
        return d

    def weakform(self, vars, mat):
        Ms = mat["Ms"]
        A = 2*mat["Aex"] * g_e / Ms
        alpha = mat["alpha_LLG"]

        wf = 0
        theta = 1
        m, test_m = vars[self.variableName]
        lam, test_lam = vars[self.variableName+"_lam"]

        # Left-hand side, normalization term
        wf += m.cross(m.t).dot(test_m)*dx + alpha*m.t.dot(test_m)*dx
        wf += (lam*m.dot(test_m) + m.t.dot(m)*test_lam)*dx

        # Exchange term
        wf += A*grad(m).ddot(grad(test_m))*dx
        wf += A*grad(m.t).ddot(grad(test_m))*theta/dti*dx
        wf += A*grad(m).ddot(grad(m))*m.t.dot(test_m)*theta/dti*dx

        for ex in self.domainConditions.get(ExternalMagneticField):
            B = mat[ex.values]
            wf += -g_e*B.dot(test_m)*dx(ex.geometries)
            wf += g_e*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(ex.geometries)

        for uni in self.domainConditions.get(UniaxialAnisotropy):
            u, Ku = mat["u_Ku/norm(u_Ku)"], mat["Ku"]
            B = 2*Ku/Ms*m.dot(u)*u
            wf += -g_e*B.dot(test_m)*dx(uni.geometries)
            wf += g_e*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(uni.geometries)

        for cu in self.domainConditions.get(CubicAnisotropy):
            Kc = mat["Kc"]
            B = -4/Ms*util.einsum("ijkl,j,k,l->i", Kc, m, m, m)
            wf += -g_e*B.dot(test_m)*dx(cu.geometries)

        for cms in self.domainConditions.get(CubicMagnetoStriction):
            Ks = mat["K_MS"]
            du = util.grad(mat[cms.values])
            e = (du + du.T)/2
            B = -2/Ms*util.einsum("ijkl,j,kl->i", Ks, m, e)
            wf += -g_e*B.dot(test_m)*dx(cms.geometries)
            #wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(cms.geometries)

        for cms in self.domainConditions.get(CubicMagnetoRotationCoupling):
            Kc = mat["Kc"]
            du = util.grad(mat[cms.values])
            w = (du.T-du)/2
            K = util.einsum("ijkl,Ii->Ijkl", Kc, w) + util.einsum("ijkl,Jj->iJkl", Kc, w) + util.einsum("ijkl,Kk->ijKl", Kc, w) + util.einsum("ijkl,Ll->ijkL", Kc, w)
            B = -4/Ms*util.einsum("ijkl,j,k,l->i", K, m, m, m)
            wf += -g_e*B.dot(test_m)*dx(cms.geometries)

        for bar in self.domainConditions.get(BarnettEffect):
            u = mat[bar.values]
            rut = util.rot(u.t)/2
            wf += -rut.dot(test_m) * dx(bar.geometries)
            #wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(bar.geometries)

        for sc in self.domainConditions.get(MagneticScalarPotential):
            phi = mat[sc.values]
            B = -mu_0*grad(phi)
            wf += -g_e*B.dot(test_m)*dx(sc.geometries)
            #wf += g*m.dot(B)*m.t.dot(test_m)*theta/dti*dx(sc.geometries)

        for st in self.domainConditions.get(SpinTransferTorque):
            beta = mat["beta_st"]
            u = -mu_B/e/Ms*mat[st.values]/(1+beta**2)
            w = u.dot(grad(m))
            wf += -(m.cross(w) + beta*w).dot(test_m)*dx(st.geometries)

        for th in self.domainConditions.get(ThermalFluctuation):
            T, R = mat[th.T], mat[th.R]
            D = alpha*k_B*T*dti/(Ms*g_e*Ve)
            B = util.sqrt(2*D)*R
            wf += -g_e*B.dot(test_m)*dx(th.geometries)

        return wf

