from .. import common, general 

    

class DirichletBoundary(common.DirichletBoundary):
    pass


class InitialCondition(common.InitialCondition):
    unit = "A"


class Source(common.Source):
    unit = "A/m^2"


class MagnetostatisticsModel(general.PoissonModel):
    className = "Magnetostatistics"
    initialConditionTypes = [InitialCondition]

