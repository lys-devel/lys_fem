from lys_fem.mf import mfem, MFEMLinearModel


class MFEMHeatConductionModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        self._mat = mat
        super().__init__(mfem.H1_FECollection(1, mesh.Dimension()), model, mesh)

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
