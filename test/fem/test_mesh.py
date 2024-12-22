import os
from lys_fem import geometry
from lys_fem.fem import FEMProject

from ..base import FEMTestCase

class parameters_test(FEMTestCase):
    @property
    def model(self):
        p = FEMProject(2)
        p.geometries.add(geometry.Rect(0, 0, 0, 1, 1))

        m = p.mesher
        model = p.geometries.generateGeometry()
        m._generate(model)
        nodes, _, _ = model.mesh.getNodes()
        return p, model, m, len(nodes)

    def test_basic(self):
        proj, model, m, nodes = self.model
        self.assertTrue(nodes > 0)

    def test_refinement(self):
        proj, model, m, nodes = self.model
        m.setRefinement(1)
        m._generate(model)
        nodes_new = len(model.mesh.getNodes()[0])
        self.assertTrue(nodes_new > nodes)

    def test_sizeConstraint(self):
        proj, model, m, nodes = self.model
        m.addSizeConstraint(geomType="Edge", geometries=[1,2], size=0.05)
        m._generate(model)
        nodes1 = len(model.mesh.getNodes()[0])
        m.addSizeConstraint(geomType="Surface", geometries=[1], size=0.02)
        m._generate(model)
        nodes2 = len(model.mesh.getNodes()[0])
        self.assertTrue(nodes1 > nodes)
        self.assertTrue(nodes2 > nodes1)

    def test_transfinite(self):
        proj, model, m, nodes = self.model
        m.addTransfinite("Surface", [1])
        m._generate(model)

    def test_file(self):
        proj, model, m, nodes = self.model
        m.file = os.path.dirname(__file__) + "/mesh/mesh_ref.msh"
        m._generate(model)
        nodes1 = len(model.mesh.getNodes()[0])
        self.assertEqual(nodes1, 842)
