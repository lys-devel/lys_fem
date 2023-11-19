import numpy as np
from lys_fem.mf import mfem, MFEMNonlinearModel


class MFEMLLGModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(mfem.H1_FECollection(1, mesh.Dimension()), model, mesh, vDim=1)
        self._mat = mat
        self._alpha = 0.0
        
        self._space_solution = mfem.FiniteElementSpace(mesh, self._fec, 3)
        self._spaces=[mfem.FiniteElementSpace(mesh, self._fec, 1, mfem.Ordering.byVDIM) for i in range(4)]
        self._block_offsets = mfem.intArray(list([0]+[sp.GetTrueVSize() for sp in self._spaces]))
        self._block_offsets.PartialSum()

        self._var = VariableMatrices(self._spaces)
        self._mass = MassMatrix(self._spaces, self._alpha, self._block_offsets)
        self._stiff = StiffnessMatrix(self._spaces, self._block_offsets)
        self._ext = ExternalMagneticField(self._spaces, self._block_offsets, mfem.VectorConstantCoefficient([0,0,1]))

        self._init = False

    def setInitialValue(self, x=None, xt=None):
        self._x1_gf = mfem.GridFunction(self._spaces[0])
        self._x1_gf.ProjectCoefficient(mfem.InnerProductCoefficient(x, mfem.VectorConstantCoefficient([1,0,0])))
        self._x2_gf = mfem.GridFunction(self._spaces[1])
        self._x2_gf.ProjectCoefficient(mfem.InnerProductCoefficient(x, mfem.VectorConstantCoefficient([0,1,0])))
        self._x3_gf = mfem.GridFunction(self._spaces[2])
        self._x3_gf.ProjectCoefficient(mfem.InnerProductCoefficient(x, mfem.VectorConstantCoefficient([0,0,1])))
        self._x4_gf = mfem.GridFunction(self._spaces[3])
        self._x4_gf.Assign(0.0)

        self._XL0 = mfem.BlockVector(self._block_offsets)
        self._x1_gf.GetTrueDofs(self._XL0.GetBlock(0))
        self._x2_gf.GetTrueDofs(self._XL0.GetBlock(1))
        self._x3_gf.GetTrueDofs(self._XL0.GetBlock(2))
        self._x4_gf.GetTrueDofs(self._XL0.GetBlock(3))

    def update(self, m):
        self._ess_tdof_list = self.essential_tdof_list()
        self._assemble_b()
        coef = self._make_coefs(m)
        self._vars = self._var.update(self._ess_tdof_list, coef)
        self._M, self._gM = self._mass.update(self._ess_tdof_list, self._vars, self.x0)
        self._K, self._gK = self._stiff.update(self._ess_tdof_list, self._vars)
        eK, egK = self._ext.update(self._ess_tdof_list)
        self._K, self._gK = self._K + eK, self._gK + egK 

    def _make_coefs(self, mv):
        m = mfem.BlockVector(mv, self._block_offsets)
        self._res_coefs = []

        self._cgf1 = mfem.GridFunction(self._spaces[0])
        self._cgf1.SetFromTrueDofs(m.GetBlock(0))
        self._c1 = mfem.GridFunctionCoefficient(self._cgf1)
        self._res_coefs.append(self._c1)

        self._cgf2 = mfem.GridFunction(self._spaces[1])
        self._cgf2.SetFromTrueDofs(m.GetBlock(1))
        self._c2 = mfem.GridFunctionCoefficient(self._cgf2)
        self._res_coefs.append(self._c2)

        self._cgf3 = mfem.GridFunction(self._spaces[2])
        self._cgf3.SetFromTrueDofs(m.GetBlock(2))
        self._c3 = mfem.GridFunctionCoefficient(self._cgf3)
        self._res_coefs.append(self._c3)

        self._cgf4 = mfem.GridFunction(self._spaces[3])
        self._cgf4.SetFromTrueDofs(m.GetBlock(3))
        self._c4 = mfem.GridFunctionCoefficient(self._cgf4)
        self._res_coefs.append(self._c4)
        return self._res_coefs

    def _assemble_b(self):
        self._B = mfem.BlockVector(self._block_offsets)

        self.b1 = mfem.LinearForm(self._spaces[0])
        self.b1.AddDomainIntegrator(mfem.DomainLFIntegrator(mfem.ConstantCoefficient(0)))
        self.b1.Assemble()
        self.b1gf = mfem.GridFunction(self._spaces[0], self.b1)
        self.b1gf.GetTrueDofs(self._B.GetBlock(0))
        self.b1gf.GetTrueDofs(self._B.GetBlock(1))
        self.b1gf.GetTrueDofs(self._B.GetBlock(2))

        self.b4 = mfem.LinearForm(self._spaces[3])
        self.b4.AddDomainIntegrator(mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1)))
        self.b4.Assemble()
        self.b4gf = mfem.GridFunction(self._spaces[3], self.b4)
        self.b4gf.GetTrueDofs(self._B.GetBlock(3))

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

        self._gf_sol = mfem.GridFunction(self._space_solution)
        self._gf_sol.SetFromTrueDofs(self._XL0[:self._block_offsets[3]])

        return self._gf_sol

    def dualToPrime(self, d):
        d = mfem.BlockVector(d, self._block_offsets)
        res = mfem.BlockVector(self._block_offsets)
        for i, sp in enumerate(self._spaces):
            m = mfem.BilinearForm(sp)
            m.AddDomainIntegrator(mfem.MassIntegrator())
            m.Assemble()
            M = mfem.SparseMatrix()
            m.FormSystemMatrix(mfem.intArray(), M)

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

class VariableMatrices:
    def __init__(self, spaces):
        self._spaces = spaces

    def update(self, ess_tdof, coefs):
        self._ess_tdof_list = ess_tdof
        self.k1 = mfem.BilinearForm(self._spaces[0])
        self.k1.AddDomainIntegrator(mfem.MassIntegrator(coefs[0]))
        self.k1.Assemble()
        self._m1 = mfem.SparseMatrix()
        self.k1.FormSystemMatrix(mfem.intArray(), self._m1)

        self.k2 = mfem.BilinearForm(self._spaces[1])
        self.k2.AddDomainIntegrator(mfem.MassIntegrator(coefs[1]))
        self.k2.Assemble()
        self._m2 = mfem.SparseMatrix()
        self.k2.FormSystemMatrix(mfem.intArray(), self._m2)

        self.k3 = mfem.BilinearForm(self._spaces[2])
        self.k3.AddDomainIntegrator(mfem.MassIntegrator(coefs[2]))
        self.k3.Assemble()
        self._m3 = mfem.SparseMatrix()
        self.k3.FormSystemMatrix(mfem.intArray(), self._m3)

        self.k4 = mfem.BilinearForm(self._spaces[3])
        self.k4.AddDomainIntegrator(mfem.MassIntegrator(coefs[3]))
        self.k4.Assemble()
        self._m4 = mfem.SparseMatrix()
        self.k4.FormSystemMatrix(mfem.intArray(), self._m4)
        return [self._m1, self._m2, self._m3, self._m4]

class MassMatrix:
    def __init__(self, spaces, alpha, offsets):
        self._spaces = spaces
        self._alpha = alpha
        self._block_offsets = offsets
        self._init = False

    def update(self, ess_tdof, coefs, x0):
        self._ess_tdof_list = ess_tdof
        if not self._init:
            self._init = True
            self._assemble_M0()
        self._assemble_M(coefs)
        self._assemble_grad_Mx(coefs, x0)
        return self._M, self._M

    def _assemble_M0(self):
        self.M1 = mfem.SparseMatrix()
        self.m1 = mfem.BilinearForm(self._spaces[0])
        self.m1.AddDomainIntegrator(mfem.MassIntegrator())
        self.m1.Assemble()
        self.m1.FormSystemMatrix(mfem.intArray(), self.M1)

    def _assemble_M(self, m):
        alpha = self._alpha
        self._M = mfem.BlockOperator(self._block_offsets)
        self._M.SetBlock(0, 0, self.M1)
        self._M.SetBlock(0, 1, alpha * m[2])
        self._M.SetBlock(0,2,m[1]*(-alpha))
                         
        self._M.SetBlock(1,1,self.M1)
        self._M.SetBlock(1,0,m[2]*(-alpha))
        self._M.SetBlock(1,2,m[0]*alpha)
                         
        self._M.SetBlock(2,2,self.M1)
        self._M.SetBlock(2,0,m[1]*alpha)
        self._M.SetBlock(2,1,m[0]*(-alpha))

    def _assemble_grad_Mx(self, m, x0):
        self._assemble_Md11(x0)
        alpha = self._alpha
        m1 = mfem.Add(2*alpha, m[0], -alpha, self._m01)
        m2 = mfem.Add(2*alpha, m[1], -alpha, self._m02)
        m3 = mfem.Add(2*alpha, m[2], -alpha, self._m03)
        m1m = mfem.Add(-2*alpha, m[0], alpha, self._m01)
        m2m = mfem.Add(-2*alpha, m[1], alpha, self._m02)
        m3m = mfem.Add(-2*alpha, m[2], alpha, self._m03)

        self._gM = mfem.BlockOperator(self._block_offsets)
        self._gM.SetBlock(0,0,self.M1)
        self._gM.SetBlock(0,1,m3)
        self._gM.SetBlock(0,2,m2m)
                         
        self._gM.SetBlock(1,1,self.M1)
        self._gM.SetBlock(1,0,m3m)
        self._gM.SetBlock(1,2,m1)
                         
        self._gM.SetBlock(2,2,self.M1)
        self._gM.SetBlock(2,0,m2)
        self._gM.SetBlock(2,1,m1m)

    def _assemble_Md11(self, m0):
        m0 = self.createM0Coefs(m0)
        self.k1 = mfem.BilinearForm(self._spaces[0])
        self.k1.AddDomainIntegrator(mfem.MassIntegrator(m0[0]))
        self.k1.Assemble()
        self._m01 = mfem.SparseMatrix()
        self.k1.FormSystemMatrix(mfem.intArray(), self._m01)

        self.k2 = mfem.BilinearForm(self._spaces[1])
        self.k2.AddDomainIntegrator(mfem.MassIntegrator(m0[1]))
        self.k2.Assemble()
        self._m02 = mfem.SparseMatrix()
        self.k2.FormSystemMatrix(mfem.intArray(), self._m02)

        self.k3 = mfem.BilinearForm(self._spaces[2])
        self.k3.AddDomainIntegrator(mfem.MassIntegrator(m0[2]))
        self.k3.Assemble()
        self._m03 = mfem.SparseMatrix()
        self.k3.FormSystemMatrix(mfem.intArray(), self._m03)

    def createM0Coefs(self, m0):
        m = mfem.BlockVector(m0, self._block_offsets)
        self._res_coefs = []

        self._cgf1 = mfem.GridFunction(self._spaces[0])
        self._cgf1.SetFromTrueDofs(m.GetBlock(0))
        self._c1 = mfem.GridFunctionCoefficient(self._cgf1)
        self._res_coefs.append(self._c1)

        self._cgf2 = mfem.GridFunction(self._spaces[1])
        self._cgf2.SetFromTrueDofs(m.GetBlock(1))
        self._c2 = mfem.GridFunctionCoefficient(self._cgf2)
        self._res_coefs.append(self._c2)

        self._cgf3 = mfem.GridFunction(self._spaces[2])
        self._cgf3.SetFromTrueDofs(m.GetBlock(2))
        self._c3 = mfem.GridFunctionCoefficient(self._cgf3)
        self._res_coefs.append(self._c3)
        return self._res_coefs
    
class StiffnessMatrix:
    def __init__(self, spaces, offsets, kLi=1e-5*0):
        self._spaces = spaces
        self._block_offsets = offsets
        self._kLi = kLi
        self._init = False

    def update(self, ess_tdof, coefs):
        self._ess_tdof_list = ess_tdof
        if not self._init:
            self._init = True
            self._assemble_K44()
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

    def _assemble_K44(self):
        self._K44 = mfem.SparseMatrix()
        self._k44 = mfem.BilinearForm(self._spaces[3])
        self._k44.AddDomainIntegrator(mfem.MassIntegrator(mfem.ConstantCoefficient(self._kLi)))
        self._k44.Assemble()
        self._k44.FormSystemMatrix(mfem.intArray(), self._K44)

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
    def __init__(self, spaces, offsets, B):
        self._spaces = spaces
        self._block_offsets = offsets
        self._B = B
        self._init = False

    def update(self, ess_tdof):
        if not self._init:
            self._init = True
            self._K = mfem.BlockOperator(self._block_offsets)
            K1, K2, K3 = self._assemble_B(self._B)
            self._K.SetBlock(0, 1,  K3)
            self._K.SetBlock(0, 2, -K2)
            self._K.SetBlock(1, 0, -K3)
            self._K.SetBlock(1, 2,  K1)
            self._K.SetBlock(2, 0,  K2)
            self._K.SetBlock(2, 1, -K1)
        return self._K, self._K
    
    def _assemble_B(self, B):
        B1 = mfem.InnerProductCoefficient(B, mfem.VectorConstantCoefficient([1,0,0]))
        B2 = mfem.InnerProductCoefficient(B, mfem.VectorConstantCoefficient([0,1,0]))
        B3 = mfem.InnerProductCoefficient(B, mfem.VectorConstantCoefficient([0,0,1]))

        K1 = mfem.SparseMatrix()
        K2 = mfem.SparseMatrix()
        K3 = mfem.SparseMatrix()
        
        self._k1 = mfem.BilinearForm(self._spaces[0])
        self._k1.AddDomainIntegrator(mfem.MassIntegrator(B1))
        self._k1.Assemble()
        self._k1.FormSystemMatrix(mfem.intArray(), K1)

        self._k2 = mfem.BilinearForm(self._spaces[0])
        self._k2.AddDomainIntegrator(mfem.MassIntegrator(B2))
        self._k2.Assemble()
        self._k2.FormSystemMatrix(mfem.intArray(), K2)

        self._k3 = mfem.BilinearForm(self._spaces[0])
        self._k3.AddDomainIntegrator(mfem.MassIntegrator(B3))
        self._k3.Assemble()
        self._k3.FormSystemMatrix(mfem.intArray(), K3)

        return K1, K2, K3