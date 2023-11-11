import numpy as np
from lys_fem.mf import mfem, MFEMNonlinearModel


class MFEMLLGModel(MFEMNonlinearModel):
    def __init__(self, fec, model, mesh, mat):
        super().__init__(fec, model, mesh)
        self._mat = mat
        self._fespace_lam = mfem.FiniteElementSpace(mesh, fec, 1, mfem.Ordering.byVDIM)

        self._lam = mfem.GridFunction(self._fespace_lam)
        self._lam.Assign(0.0)
    
        self._block_offsets = mfem.intArray([0, self.space.GetVSize(), self._fespace_lam.GetVSize()])
        self._block_offsets.PartialSum()

        self._init = False

    def update(self, m):
        if not self._init:
            self._init = True
            self._ess_tdof_list = self.essential_tdof_list()
            self._assemble_M0()
        mc = self._make_mc(m)
        self._assemble_M1(mc)
        pass

    def _make_mc(self, m):
        self.m_gf = mfem.GridFunction(self.space)
        self.m_gf.SetFromTrueDofs(m.GetBlock(0))
        self.m_c = mfem.VectorGridFunctionCoefficient(self.m_gf)
        return self.m_c

    def _assemble_M0(self):
        self._m0 = mfem.BilinearForm(self.space)
        self._m0.AddDomainIntegrator(mfem.MassIntegrator())
        self._m0.Assemble()
        self._M0 = mfem.SparseMatrix()
        self._m0.FormSystemMatrix(self._ess_tdof_list, self._M0)

    def _assemble_M1(self, m):
        self._m1 = mfem.BilinearForm(self.space)
        self._m1.AddDomainIntegrator(mfem.MixedCrossProductIntegrator(m))
        self._m1.Assemble()
        self._M1 = mfem.SparseMatrix()
        self._m1.FormSystemMatrix(self._ess_tdof_list, self._M1)

    @property
    def x0(self):
        self._x, _ = self.getInitialValue()
        self._XL0 = mfem.BlockVector(self._block_offsets)
        self._x.GetTrueDofs(self._XL0.GetBlock(0))
        self._lam.GetTrueDofs(self._XL0.GetBlock(1))
        return self._XL0

    @property
    def K(self):
        return None
    
    @property
    def b(self):
        return None