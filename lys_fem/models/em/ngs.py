from ngsolve import grad, dx, ds, x, y, z

from lys_fem.ngs import util
from .. import general


class NGSMagnetostatisticsModel(general.NGSPoissonModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, mat)

    def bilinearform(self, tnt, sols):
        return super().bilinearform(tnt, sols)
    
    def linearform(self, tnt, sols):
        return super().linearform(tnt, sols)
