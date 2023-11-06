from lys_fem.mf import mfem, MFEMLinearModel


class MFEMHeatConductionModel(MFEMLinearModel):
    def __init__(self, fec, model, mesh, mat):
        super().__init__(fec, model, mesh)
        self._mat = mat

    def assemble_m(self):
        m = mfem.BilinearForm(self.space)
        m.AddDomainIntegrator(mfem.MassIntegrator(self._mat["Heat Conduction"]["C_v"]))
        m.Assemble()
        return m

    def assemble_a(self):
        a = mfem.BilinearForm(self.space)
        a.AddDomainIntegrator(mfem.DiffusionIntegrator(self._mat["Heat Conduction"]["k"]))
        a.Assemble()
        return a
