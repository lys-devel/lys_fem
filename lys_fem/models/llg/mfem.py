import numpy as np
from lys_fem.mf import mfem, MFEMNonlinearModel


class MFEMLLGModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(mfem.ND_FECollection(1, mesh.Dimension()), model, mesh, vDim=1)
        self._mat = mat
        self._alpha = 0.0
        self._kLi = 1e-3
        self._gam = 1.760859770e11/1e12
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
        #self._test2()

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

    def _test2(self):
        class grad(mfem.PyOperator):
            def __init__(self, gK):
                super().__init__(gK.Width())
                self.gK = gK

            def Mult(self, x, res):
                res *= 0
                self.gK.AddMult(x, res)
        print("test2-------------------")
        K, b, x0 = self.K, self.b, self.x0
        g = grad(self.grad_Kx)
        sol, _ = mfem.getSolver("CG")
        sol.SetOperator(g)

        z = mfem.Vector(x0.Size())
        x1 = mfem.Vector(x0.Size())
        K.Mult(x0, z)
        z -= b
        sol.Mult(z,x1)
        x = mfem.BlockVector(x1, self._block_offsets)

        xg, _ = self.getInitialValue()
        xg.SetFromTrueDofs(x.GetBlock(0))
        res = []
        for i in range(3):
            v = mfem.Vector()
            xg.GetNodalValues(v, i+1)
            res.append(v[0])
        print("m", np.array(res))
        
        lamg = mfem.GridFunction(self._fespace_lam)
        lamg.SetFromTrueDofs(x.GetBlock(1))
        v = mfem.Vector()
        lamg.GetNodalValues(v,1)
        print("lambda", v[0])
        print()

    def _test3(self):
        dti = 1e1
        kLi = self._kLi
        lag = 0
        gB = self._gam * 1
        x0 = np.array([1,0,0])
        for ti in range(100):
            print("time", ti, x0[0]**2+x0[1]**2, x0)
            x = np.array(x0)
            for i in range(100):
                A = np.array([[dti, gB, 2*x[0]*lag], [-gB, dti, 2*x[1]*lag], [x[0]*lag, x[1]*lag, kLi]])
                b = np.array([dti*x0[0], dti*x0[1], 1])
                J = np.array([[dti-2*x[2], gB, 2*x[0]*lag], [-gB, dti-2*x[2], 2*x[1]*lag], [2*x[0]*lag, 2*x[1]*lag, kLi]])
                Axb = (A.dot(x) - b)
                x = x - np.linalg.inv(J).dot(Axb)
                #print("correct result", x, x[0]**2+x[1]**2)
            x0 = x

    def _assemble_K11(self):
        B = mfem.VectorConstantCoefficient(mfem.Vector([0, 0, -self._gam]))
        self._k11 = mfem.BilinearForm(self.space)
        #self._k11.AddDomainIntegrator(mfem.VectorFEMassIntegrator(mfem.ConstantCoefficient(2.0)))
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
        self._k21 = mfem.MixedBilinearForm(self._fespace_lam, self.space)
        self._k21.AddDomainIntegrator(mfem.MixedVectorProductIntegrator(m))
        self._k21.Assemble()
        self._K21_ptr = mfem.OperatorHandle()
        self._k21.FormRectangularSystemMatrix(self._ess_tdof_list, mfem.intArray(), self._K21_ptr)
        self._K21i = self._k21.SpMat()
        self._K21 = mfem.Transpose(self._K21i)

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
        self._assemble_Kd21(m)
        self._gK = mfem.BlockMatrix(self._block_offsets)
        #self._gK.SetBlock(0, 0, self._K11)
        self._gK.SetBlock(0, 0, self._K11 + self._Kd11)
        self._gK.SetBlock(0, 1, self._K12)
        self._gK.SetBlock(1, 0, self._Kd21)
        self._gK.SetBlock(1, 1, self._K22)
        return

        res = mfem.BlockVector(self._block_offsets)
        self.grad_Kx.GetBlock(1,0).Mult(self._vec.GetBlock(0), res.GetBlock(1))

        d = self.dualToPrime(res)
        #print("kd21 norm")
        #self.printNodalValue(d)

    def _assemble_Kd11(self, lam):
        self._l2 = mfem.ProductCoefficient(2, lam)
        self._kd11 = mfem.BilinearForm(self.space)
        self._kd11.AddDomainIntegrator(mfem.VectorFEMassIntegrator(self._l2))
        self._kd11.Assemble()
        self._Kd11 = mfem.SparseMatrix()
        self._kd11.FormSystemMatrix(self._ess_tdof_list, self._Kd11)

    def _assemble_Kd21(self, m):
        m2 = mfem.ScalarVectorProductCoefficient(2, m)
        self._kd21 = mfem.MixedBilinearForm(self.space, self._fespace_lam)
        self._kd21.AddDomainIntegrator(mfem.MixedDotProductIntegrator(m2))
        self._kd21.Assemble()
        self._Kd21_ptr = mfem.OperatorHandle()
        self._kd21.FormRectangularSystemMatrix(self._ess_tdof_list, mfem.intArray(), self._Kd21_ptr)
        self._Kd21 = self._kd21.SpMat()

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
        return

        A = mfem.SparseMatrix()
        X = mfem.Vector()
        print("form")
        self._k22.FormLinearSystem(mfem.intArray(),self._lam,b,A,X,self._B.GetBlock(1))

        solver, prec = mfem.getSolver()
        solver.SetOperator(A)
        solver.Mult(self._B.GetBlock(1), X)
        #self._k22.RecoverFEMSolution(X, b, self._lam)

        v = bgf.GetDataArray()
        #bgf.GetNodalValues(v, 1)
        print("b", v[0])
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
    def preconditioner(self):
        return None
        self._s = mfem.BilinearForm(self._fespace_lam)
        self._s.AddDomainIntegrator(mfem.MassIntegrator(mfem.ConstantCoefficient(1.0)))
        self._s.Assemble()
        self._s.Finalize()
        self._S = self._s.LoseMat()

        self._prec = JacobianPreconditioner(self._S, self._block_offsets)
        return self._prec

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
        m = np.array([v1[0], v2[0], v3[0]])
        np.set_printoptions(precision=8, suppress=True)
        print("update", np.linalg.norm(m)**2, m, v4[0], self._gam*1e-13, (self._gam*1e-13)**2)
        #print([xx for xx in self._x])

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
            ppp.append(vv[0])

        gfl = mfem.GridFunction(self._fespace_lam)
        gfl.SetFromTrueDofs(vr.GetBlock(1))
        gfl.GetNodalValues(vv, 1)
        ppp.append(vv[0])
        print(ppp)

class JacobianPreconditioner(mfem.Solver):
    def __init__(self, S, offsets):
        super().__init__(offsets[2])
        self.block_offsets = offsets
        self.gamma = 0.00001
        self.Ki, self.Ki_prec = mfem.getSolver("GMRES", prec="GS")
        self.Si, self.Si_prec = mfem.getSolver("CG", prec="GS")
        self.Si.SetOperator(S)

    def SetOperator(self, op):
        jac = mfem.Opr2BlockOpr(op)
        self.Ki.SetOperator(jac.GetBlock(0, 0))
        self.B = jac.GetBlock(0, 1)

    def Mult(self, k, y):
        # Extract the blocks from the input and output vectors
        off = self.block_offsets
        k1 = k[off[0]:off[1]]
        k2 = k[off[1]:off[2]]

        z1 = mfem.Vector(off[1]-off[0])
        z2 = mfem.Vector(off[1]-off[0])

        # Perform the block elimination for the preconditioner
        self.Si.Mult(k2, z2)
        z2 *= -self.gamma
        self.Ki.Mult(k1, z1)

        w1 = mfem.Vector(off[1]-off[0])
        w2 = mfem.Vector(off[1]-off[0])

        y1 = y[off[0]:off[1]]
        y2 = y[off[1]:off[2]]

        y2.Set(1, z2)
        self.B.Mult(z2, w1) # z1 = B y2
        self.Ki.Mult(w1, w2) # y1 = Ki (k1 - B y2)
        mfem.subtract_vector(z1, w2, y1) # z2 = k1-z1 = k1 - B y2

    def Mult2(self, k, y):
        # Extract the blocks from the input and output vectors
        off = self.block_offsets
        k1 = k[off[0]:off[1]]
        k2 = k[off[1]:off[2]]

        tx = k1
        ty1 = mfem.Vector(off[1]-off[0])
        ty2 = mfem.Vector(off[2]-off[1])

        self.Ki.Mult(k1, ty1)
        self.B.MultTranspose(ty1, ty2)
        ty2.Neg()
        ty2 += k2

        z1 = mfem.Vector(off[1]-off[0])
        z2 = mfem.Vector(off[1]-off[0])

        # Perform the block elimination for the preconditioner
        self.Si.Mult(ty2, z2)
        z2 *= self.gamma
        self.Ki.Mult(tx, z1)

        w1 = mfem.Vector(off[1]-off[0])
        w2 = mfem.Vector(off[1]-off[0])

        y1 = y[off[0]:off[1]]
        y2 = y[off[1]:off[2]]

        y2.Set(1, z2)
        self.B.Mult(z2, w1) # z1 = B y2
        self.Ki.Mult(w1, w2) # y1 = Ki (k1 - B y2)
        mfem.subtract_vector(z1, w2, y1) # z2 = k1-z1 = k1 - B y2