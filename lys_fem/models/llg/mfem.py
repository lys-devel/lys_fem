import numpy as np
from lys_fem.mf import mfem, MFEMNonlinearModel


class MFEMLLGModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(mfem.ND_FECollection(1, mesh.Dimension()), model, mesh, vDim=1)
        self._mat = mat
        self._alpha = 0
        self._kLi = 1e-5
        self._fec_lam = mfem.H1_FECollection(1, mesh.Dimension())
        self._fespace_lam = mfem.FiniteElementSpace(mesh, self._fec_lam, 1, mfem.Ordering.byVDIM)

        self._lam = mfem.GridFunction(self._fespace_lam)
        self._lam.Assign(0.0)

        self._block_offsets = mfem.intArray([0, self.space.GetVSize(), self._fespace_lam.GetVSize()])
        self._block_offsets.PartialSum()

        self._init = False
        self._initx = False

    def update(self, m):
        if not self._init:
            self._init = True
            self._ess_tdof_list = self.essential_tdof_list()
            self._assemble_M0()
            self._assemble_K22()
            self._assemble_b()
            #self._test3()
        mc, lc = self._make_coefs(m)
        self._assemble_M(mc)
        self._assemble_grad_Mx()
        self._assemble_K(mc)
        self._assemble_grad_Kx(mc, lc)

    def _make_coefs(self, m):
        m = mfem.BlockVector(m, self._block_offsets)
        self.m_gf = mfem.GridFunction(self.space)
        self.m_gf.SetFromTrueDofs(m.GetBlock(0))
        self.m_c = mfem.VectorGridFunctionCoefficient(self.m_gf)

        self.l_gf = mfem.GridFunction(self._fespace_lam)
        self.l_gf.SetFromTrueDofs(m.GetBlock(1))
        self.l_c = mfem.GridFunctionCoefficient(self.l_gf)

        self._vec=mfem.BlockVector(m, self._block_offsets)
        return self.m_c, self.l_c

    def _assemble_M(self, mc):
        self._assemble_M11(mc)

        self._M = mfem.BlockMatrix(self._block_offsets)
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

    def _assemble_K(self, m):
        self._assemble_K11()
        self._assemble_K21(m)
        self._K = mfem.BlockMatrix(self._block_offsets)
        self._K.SetBlock(0, 0, self._K11)
        self._K.SetBlock(1, 0, self._K21)
        self._K.SetBlock(0, 1, self._K12)
        self._K.SetBlock(1, 1, self._K22)

    def _test3(self):
        dti = 100
        kLi = self._kLi
        lag = 1
        B = 2*np.pi
        x0 = np.array([1,0,0])
        for ti in range(100):
            print("time", ti, x0[0]**2+x0[1]**2, x0)
            x = np.array(x0)
            for i in range(100):
                A = np.array([[dti, B, 2*x[0]*lag], [-B, dti, 2*x[1]*lag], [x[0]*lag, x[1]*lag, kLi]])
                b = np.array([dti*x0[0], dti*x0[1], 1])
                J = np.array([[dti-2*x[2], B, 2*x[0]*lag], [-B, dti-2*x[2], 2*x[1]*lag], [2*x[0]*lag, 2*x[1]*lag, kLi]])
                Axb = (A.dot(x) - b)
                x = x - np.linalg.inv(J).dot(Axb)
                #print("correct result", x, x[0]**2+x[1]**2)
            x0 = x

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
        self._K21 = self._k21.SpMat()

        self._K12 = mfem.Transpose(self._K21)
        self._K12 *= 2

    def _assemble_grad_Mx(self):
        self._assemble_Md11()
        self._gM = mfem.BlockMatrix(self._block_offsets)
        gM11 = self._Md11 + (self._M0 + self._M11 * 2)
        self._gM.SetBlock(0, 0, gM11)

    def _assemble_Md11(self):
        self.m0_gf = mfem.GridFunction(self.space)
        self.m0_gf.SetFromTrueDofs(self.x0)
        self.m0_c = mfem.VectorGridFunctionCoefficient(self.m0_gf)

        self._cmd11 = mfem.ScalarVectorProductCoefficient(self._alpha, self.m0_c)
        self._md11 = mfem.BilinearForm(self.space)
        self._md11.AddDomainIntegrator(mfem.MixedCrossProductIntegrator(self._cmd11))
        self._md11.Assemble()
        self._Md11 = mfem.SparseMatrix()
        self._md11.FormSystemMatrix(self._ess_tdof_list, self._Md11)

    def _assemble_grad_Kx(self, m, lam):
        self._assemble_Kd11(lam)
        self._gK = mfem.BlockMatrix(self._block_offsets)
        self._gK.SetBlock(0, 0, self._K11 + self._Kd11)
        self._gK.SetBlock(0, 1, self._K12)
        self._gK.SetBlock(1, 0, mfem.Transpose(self._K12))
        self._gK.SetBlock(1, 1, self._K22)


    def _assemble_Kd11(self, lam):
        self._l2 = mfem.ProductCoefficient(2, lam)
        self._kd11 = mfem.BilinearForm(self.space)
        self._kd11.AddDomainIntegrator(mfem.VectorFEMassIntegrator(self._l2))
        self._kd11.Assemble()
        self._Kd11 = mfem.SparseMatrix()
        self._kd11.FormSystemMatrix(self._ess_tdof_list, self._Kd11)


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
        #return self._M
        return self._M.CreateMonolithic()

    @property
    def grad_Mx(self):
        #return self._gM
        return self._gM.CreateMonolithic()

    @property
    def K(self):
        #return self._K
        return self._K.CreateMonolithic()

    @property
    def grad_Kx(self):
        #return self._gK
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
    def x0(self):
        if not self._initx:
            self._initx = True
            self._x, _ = self.getInitialValue()
            self._XL0 = mfem.BlockVector(self._block_offsets)
            self._x.GetTrueDofs(self._XL0.GetBlock(0))
            self._lam.GetTrueDofs(self._XL0.GetBlock(1))
        return self._XL0

    def RecoverFEMSolution(self, X):
        self._XL0 = mfem.BlockVector(X, self._block_offsets)
        self._x.SetFromTrueDofs(self._XL0.GetBlock(0))
        self._lam.SetFromTrueDofs(self._XL0.GetBlock(1))
        v1, v2, v3, v4 = mfem.Vector(), mfem.Vector(), mfem.Vector(), mfem.Vector()
        self._x.GetNodalValues(v1, 1)
        self._x.GetNodalValues(v2, 2)
        self._x.GetNodalValues(v3, 3)
        v4=self._lam.GetDataArray()
        m = np.array([v1[0], v2[0], v3[0], v4[0]])
        np.set_printoptions(precision=8, suppress=True)
        print("[LLG] Updated: norm = ", np.linalg.norm(m)**2, ", m =", m)
        return self._x

    def dualToPrime(self, d):
        res = mfem.BlockVector(self._block_offsets)
        solver, prec = mfem.getSolver("GMRES", "GS")
        M = mfem.BlockMatrix(self._block_offsets)
        M.SetBlock(0,0,self._M0)
        M.SetBlock(1,1,self._K22)
        solver.SetOperator(M.CreateMonolithic())
        solver.Mult(d, res)
        return res
    
    def printNodalValue(self, vr, index=0):
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
        print(ppp)
