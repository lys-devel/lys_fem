import numpy as np
from lys_fem.mf import mfem, util, MFEMNonlinearModel


class MFEMLLGModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        fec =mfem.H1_FECollection(1, mesh.Dimension())
        self._spaces=[mfem.FiniteElementSpace(mesh, fec, 1, mfem.Ordering.byVDIM) for i in range(4)]
        self._space_solution = mfem.FiniteElementSpace(mesh, fec, 3)
        self._block_offsets = mfem.intArray(list([0]+[sp.GetTrueVSize() for sp in self._spaces]))
        self._block_offsets.PartialSum()

        self._mat = mat
        self._alpha = 0.0
        self._initialize(model)        

    def _initialize(self, model):
        self._ess_tdof_list = self.essential_tdof_list(self._space_solution)
        c = self.generateDomainCoefficient(self._space_solution, model.initialConditions)
        self.setInitialValue(c)

        self._mass = MassMatrix(self._spaces, self._alpha, self._block_offsets, self._ess_tdof_list)
        self._stiff = StiffnessMatrix(self._spaces, self._block_offsets, self._ess_tdof_list)
        self._ext = ExternalMagneticField(self._spaces, self._ess_tdof_list, self._block_offsets, mfem.VectorConstantCoefficient([0,0,1]))

    def setInitialValue(self, x=None):
        x1_gf = mfem.GridFunction(self._spaces[0])
        x1_gf.ProjectCoefficient(mfem.InnerProductCoefficient(x, mfem.VectorConstantCoefficient([1,0,0])))
        x2_gf = mfem.GridFunction(self._spaces[1])
        x2_gf.ProjectCoefficient(mfem.InnerProductCoefficient(x, mfem.VectorConstantCoefficient([0,1,0])))
        x3_gf = mfem.GridFunction(self._spaces[2])
        x3_gf.ProjectCoefficient(mfem.InnerProductCoefficient(x, mfem.VectorConstantCoefficient([0,0,1])))
        x4_gf = mfem.GridFunction(self._spaces[3])
        x4_gf.Assign(0.0)

        self._XL0 = mfem.BlockVector(self._block_offsets)
        x1_gf.GetTrueDofs(self._XL0.GetBlock(0))
        x2_gf.GetTrueDofs(self._XL0.GetBlock(1))
        x3_gf.GetTrueDofs(self._XL0.GetBlock(2))
        x4_gf.GetTrueDofs(self._XL0.GetBlock(3))

    def getInitialValue(self):
        self._gf_sol = mfem.GridFunction(self._space_solution)
        self._gf_sol.SetFromTrueDofs(self._XL0[:self._block_offsets[3]])
        return self._gf_sol, None

    def update(self, m):
        self._assemble_b()
        m = mfem.BlockVector(m, self._block_offsets)
        coefs = [util.coefFromVector(self._spaces[i], m.GetBlock(i)) for i in range(4)]
        self._vars = [util.bilinearForm(self._spaces[i], domainInteg=mfem.MassIntegrator(coefs[i])) for i in range(4)]

        self._M, self._gM = self._mass.update(self._vars, self.x0)
        self._K, self._gK = self._stiff.update(self._vars)
        eK, egK = self._ext.update()
        self._K, self._gK = self._K + eK, self._gK + egK 

    def _assemble_b(self):
        self._B = mfem.BlockVector(self._block_offsets)

        b1 = util.linearForm(self._spaces[0], domainInteg=mfem.DomainLFIntegrator(mfem.ConstantCoefficient(0)))
        b4 = util.linearForm(self._spaces[0], domainInteg=mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1)))

        b1gf = mfem.GridFunction(self._spaces[0], b1)
        b4gf = mfem.GridFunction(self._spaces[3], b4)

        b1gf.GetTrueDofs(self._B.GetBlock(0))
        b1gf.GetTrueDofs(self._B.GetBlock(1))
        b1gf.GetTrueDofs(self._B.GetBlock(2))
        b4gf.GetTrueDofs(self._B.GetBlock(3))

    @property
    def M(self):
        return self._M

    @property
    def grad_Mx(self):
        return self._gM

    @property
    def K(self):
        return self._K

    @property
    def grad_Kx(self):
        return self._gK

    @property
    def b(self):
        return self._B

    @property
    def grad_b(self):
        return None

    @property
    def x0(self):
        return self._XL0
    
    @property
    def timeUnit(self):
        return 1.0/1.760859770e11
    
    @property
    def preconditioner(self):
        return None

    def RecoverFEMSolution(self, X):
        np.set_printoptions(precision=8, suppress=True)
        self._XL0 = mfem.BlockVector(X, self._block_offsets)
        m = self.getNodalValue(self._XL0, 2)
        mfem.print_("[LLG] Updated: norm = ", np.linalg.norm(m[:3]), ", m =", m)
        return self.solution

    @property
    def solution(self):
        self._gf_sol = mfem.GridFunction(self._space_solution)
        self._gf_sol.SetFromTrueDofs(self._XL0[:self._block_offsets[3]])
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
        self._assemble_M(coefs)
        self._assemble_grad_Mx(coefs, x0)
        return self._M, self._M

    def _assemble_M(self, m):
        alpha = self._alpha
        self._M = mfem.BlockOperator(self._block_offsets)
        self._M.SetBlock(0, 0, self.M1)
        self._M.SetBlock(0, 1, alpha * m[2])
        self._M.SetBlock(0, 2, -alpha * m[1])
                         
        self._M.SetBlock(1, 1, self.M1)
        self._M.SetBlock(1, 0, -alpha * m[2])
        self._M.SetBlock(1, 2, alpha * m[0])
                         
        self._M.SetBlock(2,2,self.M1)
        self._M.SetBlock(2,0,m[1]*alpha)
        self._M.SetBlock(2,1,m[0]*(-alpha))

    def _assemble_grad_Mx(self, m, x0):
        self._assemble_Md11(x0)
        alpha = self._alpha

        self._gM = mfem.BlockOperator(self._block_offsets)
        self._gM.SetBlock(0,0,self.M1)
        self._gM.SetBlock(0,1, 2*alpha*m[2] - alpha*self._m03)
        self._gM.SetBlock(0,2,-2*alpha*m[1] + alpha*self._m02)
                         
        self._gM.SetBlock(1,1,self.M1)
        self._gM.SetBlock(1,0,-2*alpha*m[2] + alpha*self._m03)
        self._gM.SetBlock(1,2, 2*alpha*m[0] - alpha*self._m01)
                         
        self._gM.SetBlock(2,2,self.M1)
        self._gM.SetBlock(2,0, 2*alpha*m[1] - alpha*self._m02)
        self._gM.SetBlock(2,1,-2*alpha*m[0] + alpha*self._m01)

    def _assemble_Md11(self, m0):
        m = mfem.BlockVector(m0, self._block_offsets)
        m0 = [util.coefFromVector(self._spaces[i], m.GetBlock(i)) for i in range(3)]
        self._m01 = util.bilinearForm(self._spaces[0], domainInteg=mfem.MassIntegrator(m0[0]))
        self._m02 = util.bilinearForm(self._spaces[1], domainInteg=mfem.MassIntegrator(m0[1]))
        self._m03 = util.bilinearForm(self._spaces[2], domainInteg=mfem.MassIntegrator(m0[2]))

class StiffnessMatrix:
    def __init__(self, spaces, offsets, ess_tdof, kLi=1e-5*0):
        self._spaces = spaces
        self._block_offsets = offsets
        self._ess_tdof_list = ess_tdof
        self._kLi = kLi
        self._K44 = util.bilinearForm(self._spaces[3], domainInteg=mfem.MassIntegrator(mfem.ConstantCoefficient(self._kLi)))

    def update(self, coefs):
        self._assemble_K(coefs)
        self._assemble_grad_Kx(coefs)
        return self._K, self._gK

    def _assemble_K(self, coefs):
        m1, m2, m3, _ = coefs
        self._K = mfem.BlockOperator(self._block_offsets)
        self._K.SetBlock(0, 3, 2*m1)
        self._K.SetBlock(1, 3, 2*m2)
        self._K.SetBlock(2, 3, 2*m3)
        self._K.SetBlock(3, 0, m1)
        self._K.SetBlock(3, 1, m2)
        self._K.SetBlock(3, 2, m3)
        self._K.SetBlock(3, 3, self._K44)

    def _assemble_grad_Kx(self, coefs):
        m1, m2, m3, lam = coefs
        self._gK = mfem.BlockOperator(self._block_offsets)
        self._gK.SetBlock(0, 0, 2*lam)
        self._gK.SetBlock(1, 1, 2*lam)
        self._gK.SetBlock(2, 2, 2*lam)
        self._gK.SetBlock(3, 0, 2*m1)
        self._gK.SetBlock(3, 1, 2*m2)
        self._gK.SetBlock(3, 2, 2*m3)
        self._gK.SetBlock(0, 3, 2*m1)
        self._gK.SetBlock(1, 3, 2*m2)
        self._gK.SetBlock(2, 3, 2*m3)
        self._gK.SetBlock(3, 3, self._K44)


class ExternalMagneticField:
    def __init__(self, spaces, ess_tdofs, offsets, B):
        self._spaces = spaces
        self._block_offsets = offsets
        self._B = B
        self._ess_tdof_list = ess_tdofs
        self._initialize()

    def _initialize(self):
        self._K = mfem.BlockOperator(self._block_offsets)
        K1, K2, K3 = self._assemble_B(self._B)
        self._K.SetBlock(0, 1,  K3)
        self._K.SetBlock(0, 2, -K2)
        self._K.SetBlock(1, 0, -K3)
        self._K.SetBlock(1, 2,  K1)
        self._K.SetBlock(2, 0,  K2)
        self._K.SetBlock(2, 1, -K1)

    def update(self):
        return self._K, self._K
    
    def _assemble_B(self, B):
        B1 = mfem.InnerProductCoefficient(B, mfem.VectorConstantCoefficient([1,0,0]))
        B2 = mfem.InnerProductCoefficient(B, mfem.VectorConstantCoefficient([0,1,0]))
        B3 = mfem.InnerProductCoefficient(B, mfem.VectorConstantCoefficient([0,0,1]))

        K1 = util.bilinearForm(self._spaces[0], domainInteg=mfem.MassIntegrator(B1))
        K2 = util.bilinearForm(self._spaces[0], domainInteg=mfem.MassIntegrator(B2))
        K3 = util.bilinearForm(self._spaces[0], domainInteg=mfem.MassIntegrator(B3))
        return K1, K2, K3