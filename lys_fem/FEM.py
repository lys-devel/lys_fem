import gmsh
import numpy as np
#from sfepy.discrete import Function, Functions, FieldVariable, Integral, Material, Equation, Equations, Problem
#from sfepy.terms import Term
#from sfepy.mechanics.matcoefs import stiffness_from_lame
#from sfepy.discrete.fem import Mesh, FEDomain, Field
#from sfepy.discrete.conditions import PeriodicBC, Conditions, EssentialBC
#from sfepy.solvers.nls import Newton
#from sfepy.base.base import IndexedStruct
#from sfepy.solvers.ls import ScipyDirect


class Elastic(object):
    def __init__(self, mesh):
        self._model = mesh._model
        self._model.mesh.generate(2)

    def execute(self):
        gmsh.write("test.vtk")
        return
        mesh = Mesh.from_file("test.msh")
        domain = FEDomain("domain", mesh)

        omega = domain.create_region("Omega", "all")
        surf = [Surface(self._model, tag) for _, tag in self._model.getEntities(2)]
        surfaces = [domain.create_region('Surface' + str(i), 'vertices by get_region', 'facet',
                                         functions=Functions([Function('get_region', s)])) for i, s in enumerate(surf)]

        field = Field.from_args("fu", np.float64, 'vector', omega, approx_order=2)
        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')
        integ = Integral("i", order=3)

        m = Material('m', D=stiffness_from_lame(dim=3, lam=1.0, mu=1.0))
        f = Material('f', val=[[0], [0], [0.01]])

        t1 = Term.new('dw_lin_elastic(m.D, v, u)', integ, omega, m=m, v=v, u=u)
        t2 = Term.new('dw_volume_lvf(f.val, v)', integ, omega, f=f, v=v)

        eq = Equation('balance', t1 + t2)
        eqs = Equations([eq])
        pb = Problem("elast", equations=eqs)

        fix = EssentialBC("fix_u", surfaces[4], {"u.all": 0})
        pbc1 = PeriodicBC('periodic1', [surfaces[0], surfaces[1]], {'u.all': 'u.all'}, match='match_x_line')
        pbc2 = PeriodicBC('periodic2', [surfaces[2], surfaces[3]], {'u.all': 'u.all'}, match='match_x_line')

        pb.set_bcs(ebcs=Conditions([fix]), epbcs=Conditions([pbc1, pbc2]))
        ls = ScipyDirect({})
        nls_status = IndexedStruct()
        nls = Newton({}, lin_solver=ls, status=nls_status)

        pb.set_solver(nls)

        status = IndexedStruct()
        variables = pb.solve(status=status)

        print('Nonlinear solver status:\n', nls_status)
        print('Stationary solver status:\n', status)

        pb.save_state('linear_elasticity.vtk', variables)


class Surface:
    def __init__(self, model, tag):
        self._tag = tag
        self._model = model

    def __call__(self, coords, domain=None):
        d = np.array([self._model.getClosestPoint(2, self._tag, c)[0] - c for c in coords])
        return np.nonzero(np.linalg.norm(d, axis=1) < 1e-5)[0]
