import numpy as np
from lys_fem.mf import mfem, MFEMNonlinearModel


class MFEMLLGModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(mfem.ND_FECollection(1, mesh.Dimension()), model, mesh, vDim=1)
        self._mat = mat
        self._alpha = 0.9
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
        mc, lc = self._make_coefs(m)
        self._assemble_M(mc)
        self._assemble_grad_Mx()
        self._assemble_K(mc, lc)
        self._assemble_grad_Kx()

    def _make_coefs(self, m):
        self.m_gf = mfem.GridFunction(self.space)
        self.m_gf.SetFromTrueDofs(m)
        self.m_c = mfem.VectorGridFunctionCoefficient(self.m_gf)

        #self.l_gf = mfem.GridFunction(self._fespace_lam)
        # self.l_gf.SetFromTrueDofs(m.GetBlock(1))
        #self.l_c = mfem.GridFunctionCoefficient(self.l_gf)
        return self.m_c, None

    def _assemble_M(self, mc):
        self._assemble_M11(mc)

        self._M = mfem.BlockMatrix(self._block_offsets)
        self._M.SetBlock(0, 0, self._M0 + self._M11)
        self._M = self._M0 + self._M11

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

    def _assemble_K(self, m, lam):
        self._assemble_K11(lam)
        self._assemble_K12(m)
        self._K = mfem.BlockMatrix(self._block_offsets)
        self._K.SetBlock(0, 0, self._K11)
        self._K.SetBlock(1, 0, self._K21)
        self._K.SetBlock(0, 1, self._K12)
        self._K.SetBlock(1, 1, self._K22)
        self._K = self._K11

    def _assemble_K11(self, lam):
        B = mfem.VectorConstantCoefficient(mfem.Vector([0, 0, -1]))
        #self._l2 = mfem.ProductCoefficient(-2, lam)
        self._k11 = mfem.BilinearForm(self.space)
        self._k11.AddDomainIntegrator(mfem.MixedCrossProductIntegrator(B))
        # self._k11.AddDomainIntegrator(mfem.VectorFEMassIntegrator(self._l2))
        self._k11.Assemble()
        self._K11 = mfem.SparseMatrix()
        self._k11.FormSystemMatrix(self._ess_tdof_list, self._K11)

    def _assemble_K22(self):
        self._k22 = mfem.BilinearForm(self._fespace_lam)
        self._k22.AddDomainIntegrator(mfem.MassIntegrator(mfem.ConstantCoefficient(-1 / 1e5)))
        self._k22.Assemble()
        self._K22 = mfem.SparseMatrix()
        self._k22.FormSystemMatrix(self._ess_tdof_list, self._K22)

    def _assemble_K12(self, m):
        self._2m = mfem.ScalarVectorProductCoefficient(-2, m)
        self._k21 = mfem.MixedBilinearForm(self.space, self._fespace_lam)
        self._k21.AddDomainIntegrator(mfem.MixedDotProductIntegrator(self._2m))
        self._k21.Assemble()
        self._K21_ptr = mfem.OperatorPtr()
        self._k21.FormRectangularSystemMatrix(self._ess_tdof_list, mfem.intArray(), self._K21_ptr)
        self._K21 = self._k21.SpMat()
        self._K12 = mfem.Transpose(self._K21)

    def _assemble_grad_Mx(self):
        self._assemble_Md11()
        self._gM = mfem.BlockMatrix(self._block_offsets)
        gM11 = self._Md11 + (self._M0 + self._M11 * 2)
        self._gM.SetBlock(0, 0, gM11)
        self._gM = gM11

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

    def _assemble_grad_Kx(self):
        self._gK = mfem.BlockMatrix(self._block_offsets)
        self._gK.SetBlock(0, 0, self._K11)
        self._gK.SetBlock(0, 1, self._K12)
        self._gK.SetBlock(1, 0, mfem.Add(self._K21, self._K21))
        self._gK.SetBlock(1, 1, self._K22)
        self._gK = self._K11

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
    def x0(self):
        if not self._initx:
            self._initx = True
            self._x, _ = self.getInitialValue()
            self._XL0 = mfem.BlockVector(self._block_offsets)
            self._x.GetTrueDofs(self._XL0.GetBlock(0))
            self._lam.GetTrueDofs(self._XL0.GetBlock(1))
            self._X0 = mfem.Vector()
            self._x.GetTrueDofs(self._X0)
            print("init", [self._x[0], self._x[1], self._x[10]])
        return self._X0

    def RecoverFEMSolution(self, X):
        self._X0 = X
        self._x.SetFromTrueDofs(X)
        m = np.array([self._x[10], self._x[0], self._x[1]])
        np.set_printoptions(precision=3, suppress=True)
        print("update", m / np.linalg.norm(m))
        #print([xx for xx in self._x])

        return self._x

    @property
    def b(self):
        return None

    @property
    def grad_b(self):
        return None
