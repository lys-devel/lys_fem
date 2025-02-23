from .. import general


class NGSMagnetostaticsModel(general.NGSPoissonModel):
    pass


class NGSElectrostaticsModel(general.NGSPoissonModel):
    def __init__(self, model, **kwargs):
        super().__init__(model, coef="eps_r*8.8541878128e-12", **kwargs)