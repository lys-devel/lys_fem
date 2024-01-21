from .. import common, general 

class MagnetostatisticsModel(general.PoissonModel):
    className = "Magnetostatistics"

    @classmethod
    @property
    def initialConditionTypes(cls):
        return [InitialCondition]
    

class DirichletBoundary(common.DirichletBoundary):
    pass


class InitialCondition(common.InitialCondition):
    @classmethod
    @property
    def unit(cls):
        return "A"


class Source(common.Source):
    @classmethod
    @property
    def unit(cls):
        return "A/m^2"