import numpy as np
from lys_fem.mf import mfem, MFEMNonlinearModel


class MFEMLLGModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(mfem.RT_FECollection(0, mesh.Dimension()), model, mesh, vDim=1)
        self._mat = mat
        self._alpha = 0
        self._fec_lam = mfem.H1_FECollection(1, mesh.Dimension())
        self._fespace_lam = mfem.FiniteElementSpace(mesh, self._fec_lam, 1, mfem.Ordering.byVDIM)

        self._block_offsets = mfem.intArray([0, self.space.GetTrueVSize(), self._fespace_lam.GetTrueVSize()])
        self._block_offsets.PartialSum()

        self._mass = MassMatrix(self.space, self._alpha, self._block_offsets)
        self._stiff = StiffnessMatrix(self.space, self._fespace_lam, self._block_offsets)

        self._init = False
        self._initx = False

    def update(self, m):
        if not self._init:
            self._init = True
            self._ess_tdof_list = self.essential_tdof_list()
            self._assemble_b()
            #self._test3()
        mc, lc = self._make_coefs(m)
        self._M, self._gM = self._mass.update(self._ess_tdof_list, mc, self.x0)
        self._K, self._gK = self._stiff.update(self._ess_tdof_list, mc, lc)

    def _make_coefs(self, mv):
        m = mfem.BlockVector(mv, self._block_offsets)
        m_gf = mfem.GridFunction(self.space)
        m_gf.SetFromTrueDofs(m.GetBlock(0))
        self.m_c = mfem.VectorGridFunctionCoefficient(m_gf)

        self.l_gf = mfem.GridFunction(self._fespace_lam)
        self.l_gf.SetFromTrueDofs(m.GetBlock(1))
        self.l_c = mfem.GridFunctionCoefficient(self.l_gf)
        self.l_c = mfem.GridFunctionCoefficient(self._lam)
        return self.m_c, self.l_c

    def _assemble_b(self):
        self._B = mfem.BlockVector(self._block_offsets)

        bm = mfem.LinearForm(self.space)
        bm.AddDomainIntegrator(mfem.VectorFEDomainLFIntegrator(mfem.VectorConstantCoefficient([0,0,0])))
        bm.Assemble()
        bmgf = mfem.GridFunction(self.space, bm)
        bmgf.GetTrueDofs(self._B.GetBlock(0))

        b = mfem.LinearForm(self._fespace_lam)
        b.AddDomainIntegrator(mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1.0)))
        b.Assemble()

        bgf = mfem.GridFunction(self._fespace_lam, b)
        bgf.GetTrueDofs(self._B.GetBlock(1))

    @property
    def M(self):
        return self._M
        return self._M.CreateMonolithic()

    @property
    def grad_Mx(self):
        return self._gM
        return self._gM.CreateMonolithic()

    @property
    def K(self):
        return self._K
        return self._K.CreateMonolithic()

    @property
    def grad_Kx(self):
        return self._gK
        return self._gK.CreateMonolithic()

    @property
    def b(self):
        return self._B

    @property
    def grad_b(self):
        return None

    @property
    def timeUnit(self):
        return 1.0/1.760859770e11
    
    @property
    def preconditioner(self):
        return None
        self.op = mfem.BlockDiagonalPreconditioner(self._block_offsets)
        gM = self._mass._Md11 + (self._mass._M0 + self._mass._M11 * 2)
        gK = self._stiff._K11 + self._stiff._Kd11
        self._A1=mfem.Add(200, gM, 1, gK)
        self._cg1 = mfem.HypreSmoother(self._A1, 16)
        self.op.SetDiagonalBlock(0, self._cg1)
        return self.op

    @property
    def x0(self):
        if not self._initx:
            self._initx = True
            self._x, _ = self.getInitialValue()
            self._lam = mfem.GridFunction(self._fespace_lam)
            self._lam.Assign(0.0)

            self._XL0 = mfem.BlockVector(self._block_offsets)
            self._x.GetTrueDofs(self._XL0.GetBlock(0))
            self._lam.GetTrueDofs(self._XL0.GetBlock(1))
        return self._XL0

    def RecoverFEMSolution(self, X):
        np.set_printoptions(precision=8, suppress=True)
        self._XL0 = mfem.BlockVector(X, self._block_offsets)
        self._x.SetFromTrueDofs(self._XL0.GetBlock(0))
        m = self.getNodalValue(self._XL0, 0)
        print("[LLG] Updated: norm = ", np.linalg.norm(m[:2]), ", m =", m)
        return self._x

    def dualToPrime(self, d):
        m0 = mfem.BilinearForm(self.space)
        m0.AddDomainIntegrator(mfem.VectorFEMassIntegrator())
        m0.Assemble()
        M0 = mfem.SparseMatrix()
        m0.FormSystemMatrix(mfem.intArray(), M0)

        m1 = mfem.BilinearForm(self._fespace_lam)
        m1.AddDomainIntegrator(mfem.MassIntegrator())
        m1.Assemble()
        M1 = mfem.SparseMatrix()
        m1.FormSystemMatrix(mfem.intArray(), M1)

        d = mfem.BlockVector(d, self._block_offsets)
        res = mfem.BlockVector(self._block_offsets)

        solver1, prec1 = mfem.getSolver("CG", "GS")
        solver1.SetOperator(M0)
        solver1.Mult(d.GetBlock(0), res.GetBlock(0))

        solver2, prec2 = mfem.getSolver("CG", "GS")
        solver2.SetOperator(M1)
        solver2.Mult(d.GetBlock(1), res.GetBlock(1))
        return res
    
    def getNodalValue(self, vr, index=0):
        vv = mfem.Vector()
        ppp = []

        gfm = mfem.GridFunction(self.space)
        gfm.SetFromTrueDofs(vr.GetBlock(0))
        for d in range(3):
            gfm.GetNodalValues(vv,d+1)
            ppp.append(vv[index])

        gfl = mfem.GridFunction(self._fespace_lam)
        gfl.SetFromTrueDofs(vr.GetBlock(1))
        gfl.GetNodalValues(vv, 1)
        ppp.append(vv[index])
        return np.array(ppp)

class MassMatrix:
    def __init__(self, space, alpha, offsets):
        self.space = space
        self._alpha = alpha
        self._block_offsets = offsets
        self._init = False

    def update(self, ess_tdof, m, x0):
        self._ess_tdof_list = ess_tdof
        if not self._init:
            self._init = True
            self._assemble_M0()
        self._assemble_M(m)
        self._assemble_grad_Mx(x0)
        return self._M, self._gM

    def _assemble_M(self, mc):
        self._assemble_M11(mc)

        self._M = mfem.BlockOperator(self._block_offsets)
        self._M.SetBlock(0, 0, self._M0 + self._M11)

    def _assemble_M0(self):
        self._m0 = mfem.BilinearForm(self.space)
        self._m0.AddDomainIntegrator(mfem.VectorFEMassIntegrator())
        self._m0.Assemble()
        self._M0 = mfem.SparseMatrix()
        self._m0.FormSystemMatrix(self._ess_tdof_list, self._M0)

    def _assemble_M11(self, m):
        self._cm11 = mfem.ScalarVectorProductCoefficient(-self._alpha, m)
        self._m11 = mfem.BilinearForm(self.space)
        self._m11.AddDomainIntegrator(mfem.MixedCrossProductIntegrator(self._cm11))
        self._m11.Assemble()
        self._M11 = mfem.SparseMatrix()
        self._m11.FormSystemMatrix(self._ess_tdof_list, self._M11)

    def _assemble_grad_Mx(self, x0):
        self._assemble_Md11(x0)
        self._gM = mfem.BlockOperator(self._block_offsets)
        gM11 = self._Md11 + (self._M0 + self._M11 * 2)
        self._gM.SetBlock(0, 0, gM11)

    def _assemble_Md11(self, x0):
        self.m0_gf = mfem.GridFunction(self.space)
        self.m0_gf.SetFromTrueDofs(x0)
        self.m0_c = mfem.VectorGridFunctionCoefficient(self.m0_gf)

        self._cmd11 = mfem.ScalarVectorProductCoefficient(self._alpha, self.m0_c)
        self._md11 = mfem.BilinearForm(self.space)
        self._md11.AddDomainIntegrator(mfem.MixedCrossProductIntegrator(self._cmd11))
        self._md11.Assemble()
        self._Md11 = mfem.SparseMatrix()
        self._md11.FormSystemMatrix(self._ess_tdof_list, self._Md11)

class StiffnessMatrix:
    def __init__(self, space, space_lam, offsets, kLi=1e-8):
        self.space = space
        self._block_offsets = offsets
        self._fespace_lam = space_lam
        self._kLi = kLi
        self._init = False

    def update(self, ess_tdof, m, lam):
        self._ess_tdof_list = ess_tdof
        if not self._init:
            self._init = True
            self._assemble_K22()
        self._assemble_K(m)
        self._assemble_grad_Kx(m, lam)
        return self._K, self._gK

    def _assemble_K(self, m):
        self._assemble_K11()
        self._assemble_K21(m)
        self._K = mfem.BlockOperator(self._block_offsets)
        self._K.SetBlock(0, 0, self._K11)
        self._K.SetBlock(1, 0, self._K21)
        self._K.SetBlock(0, 1, self._K12)
        self._K.SetBlock(1, 1, self._K22)

    def _assemble_K11(self):
        B = mfem.VectorConstantCoefficient(mfem.Vector([0, 0, -1]))
        self._k11 = mfem.BilinearForm(self.space)
        self._k11.AddDomainIntegrator(mfem.MixedCrossProductIntegrator(B))
        self._k11.Assemble()
        self._K11 = mfem.SparseMatrix()
        self._k11.FormSystemMatrix(self._ess_tdof_list, self._K11)

    def _assemble_K22(self):
        self._k22 = mfem.BilinearForm(self._fespace_lam)
        self._k22.AddDomainIntegrator(mfem.MassIntegrator(mfem.ConstantCoefficient(self._kLi)))
        self._k22.Assemble()
        self._K22 = mfem.SparseMatrix()
        self._k22.FormSystemMatrix(mfem.intArray(), self._K22)

    def _assemble_K21(self, m):
        self._k21 = mfem.MixedBilinearForm(self.space, self._fespace_lam)
        self._k21.AddDomainIntegrator(mfem.MixedDotProductIntegrator(m))
        self._k21.Assemble()
        self._K21_ptr = mfem.OperatorHandle()
        self._k21.FormRectangularSystemMatrix(self._ess_tdof_list, mfem.intArray(), self._K21_ptr)
        self._K21 = self._K21_ptr.Ptr()
        self._K21_2 = mfem.MulOperator(self._K21, 2)
        self._K12 = mfem.TransposeOperator(self._K21_2)

    def _assemble_grad_Kx(self, m, lam):
        self._assemble_Kd11(lam)
        self._gK = mfem.BlockOperator(self._block_offsets)
        self._gK.SetBlock(0, 0, self._K11 + self._Kd11)
        self._gK.SetBlock(0, 1, self._K12)
        self._gK.SetBlock(1, 0, self._K21_2)
        self._gK.SetBlock(1, 1, self._K22)

    def _assemble_Kd11(self, lam):
        self._l2 = mfem.ProductCoefficient(2, lam)
        self._kd11 = mfem.BilinearForm(self.space)
        self._kd11.AddDomainIntegrator(mfem.VectorFEMassIntegrator(self._l2))
        self._kd11.Assemble()
        self._Kd11 = mfem.SparseMatrix()
        self._kd11.FormSystemMatrix(self._ess_tdof_list, self._Kd11)
