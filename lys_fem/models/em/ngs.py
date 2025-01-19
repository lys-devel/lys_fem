from .. import general


class NGSMagnetostaticsModel(general.NGSPoissonModel):
    pass


class NGSElectrostaticsModel(general.NGSPoissonModel):
    def __init__(self, model, vars, **kwargs):
        super().__init__(model, vars, coef="eps_r*8.8541878128e-12", **kwargs)