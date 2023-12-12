import sympy as sp
from lys_fem.fem import Source
from ngsolve import H1, grad, dx


class MFEMPoissonModel(MFEMModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._mesh = mesh
        self._mat = mat
        self._model = model

        self._fes = H1(mesh, definedon=[1])
        
        self._mesh1 = mfem.SubMesh.CreateFromDomain(mesh, mfem.intArray([1]))
        self._mesh2 = mfem.SubMesh.CreateFromDomain(mesh, mfem.intArray([2]))
        self._mesh3 = mfem.SubMesh.CreateFromBoundary(mesh, mfem.intArray([1]))
        self._phi1 = weakform.TrialFunction("phi1", self._mesh1, self.dirichletCondition[0], util.generateDomainCoefficient(self._mesh1, model.initialConditions, 0))
        self._phi2 = weakform.TrialFunction("phi2", self._mesh2, self.dirichletCondition[0], util.generateDomainCoefficient(self._mesh2, model.initialConditions, 0))
        self._lam1 = weakform.TrialFunction("lam_phi1", self._mesh3, [], mfem.generateCoefficient(0))

    @property
    def trialFunctions(self):
        return [self._phi1, self._phi2, self._lam1]

    @property
    def weakform(self):
        u1 = self._phi1
        v1 = weakform.TestFunction(u1)
        gu1, gv1 = grad(u1), grad(v1)

        u2 = self._phi2
        v2 = weakform.TestFunction(u2)
        gu2, gv2 = grad(u2), grad(v2)

        lam1 = self._lam1
        lv = weakform.TestFunction(lam1)

        dS1 = self._dS1

        wf = (gu1.dot(gv1)) * dV

        if self._model.domainConditions.have(Source):
            f1 = sp.Symbol("f1_poisson")
            wf += - f1 * v1 * dV
        
        wf += (gu2.dot(gv2)) * dV
        #wf += lam1*lv*dS1 - 2*lv*dV
        wf += (lam1*(v1-v2) + (u1-u2)*lv)*dS1

        return wf

    @property
    def coefficient(self):
        coefs = {}
        if self._model.domainConditions.have(Source):
            source = self._model.domainConditions.get(Source)
            coefs["f1_poisson"] = util.generateDomainCoefficient(self._mesh1, source, default=0)
        return coefs

    @property
    def integrals(self):
        return [self._dS1]
    
class MFEMPoissonModel2(MFEMModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._mesh = mesh
        self._mat = mat
        self._model = model
        self._phi = weakform.TrialFunction("phi", mesh, self.dirichletCondition[0], util.generateDomainCoefficient(mesh, model.initialConditions, 0))

    @property
    def trialFunctions(self):
        return [self._phi]

    @property
    def weakform(self):
        u = self._phi
        v = weakform.TestFunction(u)
        gu, gv = grad(u), grad(v)

        wf = (gu.dot(gv)) * dV

        if self._model.domainConditions.have(Source):
            f = sp.Symbol("f_poisson")
            wf += - f * v * dV
            pass

        return wf

    @property
    def coefficient(self):
        coefs = {}
        if self._model.domainConditions.have(Source):
            source = self._model.domainConditions.get(Source)
            coefs["f_poisson"] = util.generateDomainCoefficient(self._mesh, source, default=0)
        return coefs
    