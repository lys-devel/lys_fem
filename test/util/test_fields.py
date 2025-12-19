import numpy as np

from numpy.testing import assert_almost_equal
from lys_fem import geometry
from lys_fem import util

from ..base import FEMTestCase

class test_util(FEMTestCase):
    def _make_mesh(self, refine=0):
        mesh = geometry.GmshMesh([geometry.Line(0,0,0,1,0,0), geometry.Line(1,0,0,2,0,0)], refine=refine)
        return util.Mesh(mesh)
    
    def test_error(self):
        m = self._make_mesh()

        fs = util.H1("H1", dirichlet=[[1,3]], order=1, isScalar=True)
        u, v = fs.trial, fs.test
        wf = u * util.grad(u).dot(util.grad(v))*util.dx

        fes = util.FiniteElementSpace(fs, m)
        g = fes.gridFunction([util.x])

        # Solve nonlinear Poisson equation on given finite element space.
        util.Solver(fes, wf, linear={"solver": "pardiso"}).solve(g)
        f = util.GridField(g, fs)
        gf = f.error()

        x_list = np.linspace(0,2,7)
        err = gf(fes, x_list)

        assert_almost_equal(err, [5.71909584e-02, 5.71909584e-02, 4.93297009e-02, 1.16153092e-03, 3.19122990e-04, 1.34545672e-04, 8.54751546e-05])
