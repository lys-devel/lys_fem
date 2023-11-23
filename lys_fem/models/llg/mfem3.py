import numpy as np
from lys_fem.mf import mfem, util, MFEMNonlinearModel


class MFEMLLGModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        fec =mfem.H1_FECollection(1, mesh.Dimension())
        self._spaces = [mfem.FiniteElementSpace(mesh, fec, 3), mfem.FiniteElementSpace(mesh, fec, 1)]

        self._block_offsets = mfem.intArray(list([0]+[sp.GetTrueVSize() for sp in self._spaces]))
        self._block_offsets.PartialSum()

        self._mat = mat
        self._alpha = 0.0
        self._initialize(model)        

    def _initialize(self, model):
        self._ess_tdof_list = self.essential_tdof_list(self._spaces[0])
        c = util.generateDomainCoefficient(self._spaces[0], model.initialConditions)
        self.setInitialValue(c)

        self._mass = MassMatrix(self._spaces, self._alpha, self._block_offsets, self._ess_tdof_list)
        self._stiff = StiffnessMatrix(self._spaces, self._block_offsets, self._ess_tdof_list)
        self._ext = ExternalMagneticField(self._spaces, self._ess_tdof_list, self._block_offsets, mfem.VectorConstantCoefficient([0,0,1]))

    def setInitialValue(self, x=None):
        x1_gf = mfem.GridFunction(self._spaces[0])
        x1_gf.ProjectCoefficient(x)
 
        x2_gf = mfem.GridFunction(self._spaces[1])
        x2_gf.Assign(0.0)

        self.x0 = mfem.BlockVector(self._block_offsets)
        x1_gf.GetTrueDofs(self.x0.GetBlock(0))
        x2_gf.GetTrueDofs(self.x0.GetBlock(1))

    def update(self, m):
        self._assemble_b()
        m = mfem.BlockVector(m, self._block_offsets)
        coefs = [util.coefFromVector(self._spaces[i], m.GetBlock(i)) for i in range(2)]

        self.M, self.grad_Mx = self._mass.update(coefs, self.x0)
        self.K, self.grad_Kx = self._stiff.update(coefs)
        eK, egK = self._ext.update()
        self.K, self.grad_Kx = self.K + eK, self.grad_Kx + egK 

    def _assemble_b(self):
        self.b = mfem.BlockVector(self._block_offsets)

        b1 = util.linearForm(self._spaces[0], domainInteg=mfem.DomainLFIntegrator(mfem.ConstantCoefficient(0)))
        b2 = util.linearForm(self._spaces[1], domainInteg=mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1)))

        b1gf = mfem.GridFunction(self._spaces[0], b1)
        b2gf = mfem.GridFunction(self._spaces[1], b2)

        b1gf.GetTrueDofs(self.b.GetBlock(0))
        b2gf.GetTrueDofs(self.b.GetBlock(1))
    
    @property
    def timeUnit(self):
        return 1.0/1.760859770e11
    
    @property
    def preconditioner(self):
        return None

    def RecoverFEMSolution(self, X):
        np.set_printoptions(precision=8, suppress=True)
        self.x0 = mfem.BlockVector(X, self._block_offsets)
        m = self.getNodalValue(self.x0, 2)
        mfem.print_("[LLG] Updated: norm = ", np.linalg.norm(m[:3]), ", m =", m)
        return self.solution

    @property
    def solution(self):
        self._gf_sol = mfem.GridFunction(self._spaces[0])
        self._gf_sol.SetFromTrueDofs(self.x0[:self._block_offsets[1]])
        return self._gf_sol

    def dualToPrime(self, d):
        d = mfem.BlockVector(d, self._block_offsets)
        res = mfem.BlockVector(self._block_offsets)
        for i, sp in enumerate(self._spaces):
            M = util.bilinearForm(sp, domainInteg=mfem.MassIntegrator())
            solver, prec = mfem.getSolver("CG", "GS")
            solver.SetOperator(M)
            solver.Mult(d.GetBlock(i), res.GetBlock(i))
        return res
    
    def getNodalValue(self, vr, index=0):
        vv = mfem.Vector()
        ppp = []
        for i, sp in enumerate(self._spaces):
            gf = mfem.GridFunction(sp)
            gf.SetFromTrueDofs(vr.GetBlock(i))
            gf.GetNodalValues(vv,1)
            ppp.append(vv[index])
        return np.array(ppp)


class MassMatrix:
    def __init__(self, spaces, alpha, offsets, ess_tdof_list):
        self._spaces = spaces
        self._alpha = alpha
        self._block_offsets = offsets
        self._ess_tdof_list = ess_tdof_list
        self.M1 = util.bilinearForm(self._spaces[0], domainInteg=mfem.MassIntegrator())

    def update(self, coefs, x0):
        m, lam = coefs
        print(m)
        self._assemble_M(m)
        self._assemble_grad_Mx(m, x0)
        return self._M, self._M

    def _assemble_M(self, m):
        coef=outerProductMatrixCoefficient(mfem.ScalarVectorProductCoefficient(self._alpha, m))
        self._M0 = util.bilinearForm(self._spaces[0], domainInteg=[mfem.VectorMassIntegrator(), mfem.VectorMassIntegrator(coef)])
        self._M = mfem.BlockOperator(self._block_offsets)
        self._M.SetBlock(0,0,self._M0)

    def _assemble_grad_Mx(self, m, m0):
        m0 = mfem.BlockVector(m0, self._block_offsets)
        m0 = util.coefFromVector(self._spaces[0], m0.GetBlock(0))
        am0 = outerProductMatrixCoefficient(mfem.ScalarVectorProductCoefficient(-self._alpha, m0))
        am2 = outerProductMatrixCoefficient(mfem.ScalarVectorProductCoefficient(2*self._alpha, m))

        self.b = util.bilinearForm(self._spaces[0], domainInteg=[mfem.VectorMassIntegrator(am0), mfem.VectorMassIntegrator(am2)])

        self._gM = mfem.BlockOperator(self._block_offsets)
        self._gM.SetBlock(0,0,self.b)

class StiffnessMatrix:
    def __init__(self, spaces, offsets, ess_tdof, kLi=1e-5*0):
        self._spaces = spaces
        self._block_offsets = offsets
        self._ess_tdof_list = ess_tdof
        self._kLi = kLi
        self._K44 = util.bilinearForm(self._spaces[1], domainInteg=mfem.MassIntegrator(mfem.ConstantCoefficient(self._kLi)))

    def update(self, coefs):
        self._assemble_K(coefs)
        self._assemble_grad_Kx(coefs)
        return self._K, self._gK

    def _assemble_K(self, coefs):
        m, _ = coefs

        self._K = mfem.BlockOperator(self._block_offsets)
        v1 = util.mixedBilinearForm(*self._spaces, domainInteg=mfem.VectorMassIntegrator(mfem.ScalarVectorProductCoefficient(1, m)))
        v2 = util.mixedBilinearForm(*self._spaces, domainInteg=mfem.MixedVectorProductIntegrator(mfem.ScalarVectorProductCoefficient(2, m)))
        self._K.SetBlock(0, 1, v2)
        self._K.SetBlock(1, 0, v1)
        self._K.SetBlock(1, 1, self._K44)

    def _assemble_grad_Kx(self, coefs):
        m, lam = coefs
        l2 = util.bilinearForm(self._spaces[0], domainInteg=mfem.MassIntegrator(mfem.ProductCoefficient(2,lam)))
        m2 = util.mixedBilinearForm(*self._spaces, domainInteg=mfem.MixedVectorProductIntegrator(mfem.ScalarVectorProductCoefficient(2, m)))
        self._gK = mfem.BlockOperator(self._block_offsets)
        self._gK.SetBlock(0, 0, l2)
        self._gK.SetBlock(0, 1, m2)
        self._gK.SetBlock(1, 0, mfem.Transpose(m2))
        self._gK.SetBlock(1, 1, self._K44)


class ExternalMagneticField:
    def __init__(self, spaces, ess_tdofs, offsets, B):
        c = outerProductMatrixCoefficient(B)
        self._K1 = util.bilinearForm(spaces[0], domainInteg=mfem.VectorMassIntegrator(c))
        self._K = mfem.BlockOperator(offsets)
        self._K.SetBlock(0,0,self._K1)

    def update(self):
        return self._K, self._K
    

def outerProductMatrixCoefficient(cv):
    cx = mfem.InnerProductCoefficient(cv, mfem.VectorConstantCoefficient([1,0,0]))
    cy = mfem.InnerProductCoefficient(cv, mfem.VectorConstantCoefficient([0,1,0]))
    cz = mfem.InnerProductCoefficient(cv, mfem.VectorConstantCoefficient([0,0,1]))

    coef=mfem.MatrixArrayCoefficient(3)
    coef.Set(0, 1, cz)
    coef.Set(0, 2, mfem.ProductCoefficient(-1,cy))
    coef.Set(1, 0, mfem.ProductCoefficient(-1,cz))
    coef.Set(1, 2, cx)
    coef.Set(2, 0, cy)
    coef.Set(2, 1, mfem.ProductCoefficient(-1,cx))
    coef._cv = [cx,cy,cz]
    return coef