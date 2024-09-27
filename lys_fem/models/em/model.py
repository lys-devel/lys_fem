from .. import general 


class MagnetostaticsModel(general.PoissonModel):
    className = "Magnetostatics"


class ElectrostaticsModel(general.PoissonModel):
    className = "Electrostatics"
