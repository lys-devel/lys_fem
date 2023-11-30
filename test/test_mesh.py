from lys_fem.mf.mesh import _createMFEMMesh

from .base import FEMTestCase


class mesh_test(FEMTestCase):
    def test_mesh_1d(self):
        file = self.generateSimpleGeometry(1)
        mesh, _ = _createMFEMMesh(file)
        self.assert_array_almost_equal([x for x in mesh.attributes], [1])
        self.assert_array_almost_equal([x for x in mesh.bdr_attributes], [1,2])

    def test_mesh_2d(self):
        file = self.generateSimpleGeometry(2)
        mesh, _ = _createMFEMMesh(file)
        self.assert_array_almost_equal([x for x in mesh.attributes], [1])
        self.assert_array_almost_equal([x for x in mesh.bdr_attributes], [1,2,3,4])

    def test_mesh_3d(self):
        file = self.generateSimpleGeometry(3)
        mesh, _ = _createMFEMMesh(file)
        self.assert_array_almost_equal([x for x in mesh.attributes], [1])
        self.assert_array_almost_equal([x for x in mesh.bdr_attributes], [1,2,3,4,5,6])

