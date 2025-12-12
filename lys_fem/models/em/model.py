from .. import general 


class MagnetostaticsModel(general.PoissonModel):
    className = "Magnetostatics"


class ElectrostaticsModel(general.PoissonModel):
    className = "Electrostatics"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, k = "eps_r*eps_0", **kwargs)
