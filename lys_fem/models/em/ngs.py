from .. import general


class NGSMagnetostaticsModel(general.NGSPoissonModel):
    pass


class NGSElectrostaticsModel(general.NGSPoissonModel):
    def __init__(self, model, mesh):
        super().__init__(model, mesh, coef="eps")