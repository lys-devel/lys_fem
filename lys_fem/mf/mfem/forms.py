from . import mfem_orig

if mfem_orig.isParallel():

    class MFEMLinearForm(mfem_orig.ParLinearForm):
        def GetTrueDofs(self, vector):
            self.ParallelAssemble(vector)

    class MFEMBilinearForm(mfem_orig.ParBilinearForm):
        def SpMat(self):
            return self.ParallelAssemble()
        
    class MFEMMixedBilinearForm(mfem_orig.ParMixedBilinearForm):
        def SpMat(self):
            return self.ParallelAssemble()

else:

    class MFEMLinearForm(mfem_orig.LinearForm):
        def GetTrueDofs(self, vector):
            gf = mfem_orig.GridFunction(self.FESpace(), self)
            gf.GetTrueDofs(vector)

    MFEMBilinearForm = mfem_orig.BilinearForm
    MFEMMixedBilinearForm = mfem_orig.MixedBilinearForm