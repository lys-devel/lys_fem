from .. import common, general 


class MagnetostaticInitialCondition(common.InitialCondition):
    unit = "A"


class MagnetostaticSource(common.Source):
    unit = "A/m^2"


class MagnetostaticsModel(general.PoissonModel):
    className = "Magnetostatics"
    initialConditionTypes = [MagnetostaticInitialCondition]


class ElectrostaticInitialCondition(common.InitialCondition):
    unit = "V"


class ElectrostaticSource(common.Source):
    unit = "C/m^3"


class ElectrostaticsModel(general.PoissonModel):
    className = "Electrostatics"
    initialConditionTypes = [ElectrostaticInitialCondition]