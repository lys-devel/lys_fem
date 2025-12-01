from .. import general 


class MagnetostaticsModel(general.PoissonModel):
    className = "Magnetostatics"


class ElectrostaticsModel(general.PoissonModel):
    className = "Electrostatics"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, coef = "eps_r*8.8541878128e-12", **kwargs)
