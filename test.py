import numpy as np
import sympy as sp

from numpy.testing import assert_array_almost_equal

from lys_fem import geometry
from lys_fem.fem import FEMProject, StationarySolver, FEMSolution
from lys_fem.models import test

p = FEMProject(2)

# geometry
p.geometries.add(geometry.Rect(0, 0, 0, 1, 1))

# model: boundary and initial conditions
x = sp.Symbol("x")
model = test.NonlinearTestModel(order=1)
model.boundaryConditions.append(test.DirichletBoundary([True], geometries="all"))
model.initialConditions.append(test.InitialCondition(x, geometries="all"))
p.models.append(model)

# solver
stationary = StationarySolver()
stationary.setAdaptiveMeshRefinement("X", 1500)
p.solvers.append(stationary)

# solve
p.run()

# solution
#sol = FEMSolution()
#res = sol.eval("X", data_number=-1)
#for w in res:
#    assert_array_almost_equal(w.data, np.sqrt(2 * w.x[:, 0]), decimal=2)
#c = np.array([0.5,0.6,0.7])
#assert_array_almost_equal(sol.eval("X", data_number=-1, coords=c), np.sqrt(2*c))
