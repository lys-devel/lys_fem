import unittest
import os
import shutil
import gmsh

from lys_fem import mf
from numpy.testing import assert_array_almost_equal


class FEMTestCase(unittest.TestCase):
    path = "test/run"

    def setUp(self):
        os.makedirs(self.path, exist_ok=True)
        self._cwd = os.getcwd()
        os.chdir(self.path)
        mf.mfem.wait()

    def tearDown(self):
        mf.mfem.wait()
        os.chdir(self._cwd)
        if mf.mfem.isRoot:
            shutil.rmtree(self.path)

    def assert_array_almost_equal(self, *args, **kwargs):
        assert_array_almost_equal(*args, **kwargs)

    def generateSimpleGeometry(self, dim=1, file="mesh.msh"):
        model = gmsh.model()
        model.add("Default")
        model.setCurrent("Default")
        if dim == 1:
            p1t = model.occ.addPoint(0,0,0)
            p2t = model.occ.addPoint(1,0,0)
            model.occ.addLine(p1t, p2t)
        elif dim == 2:
            model.occ.addRectangle(0,0,0,1,1)
        else:
            model.occ.addBox(0,0,0,1,1,1)
        model.occ.removeAllDuplicates()
        model.occ.synchronize()
        for i, obj in enumerate(model.getEntities(3)):
            model.add_physical_group(dim=3, tags=[obj[1]], tag=i + 1)
            model.setPhysicalName(dim=3, tag=i+1, name=str(i+1))
        for i, obj in enumerate(model.getEntities(2)):
            model.add_physical_group(dim=2, tags=[obj[1]], tag=i + 1)
            model.setPhysicalName(dim=2, tag=i+1, name=str(i+1))
        for i, obj in enumerate(model.getEntities(1)):
            model.add_physical_group(dim=1, tags=[obj[1]], tag=i + 1)
            model.setPhysicalName(dim=1, tag=i+1, name=str(i+1))
        for i, obj in enumerate(model.getEntities(0)):
            model.add_physical_group(dim=0, tags=[obj[1]], tag=i + 1)
            model.setPhysicalName(dim=0, tag=i+1, name=str(i+1))
        model.mesh.setTransfiniteAutomatic()
        model.mesh.generate()
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(file)
        return file

    def generateSimpleMesh(self, dim):
        from lys_fem.mf.mesh import _createMFEMMesh
        file = self.generateSimpleGeometry(dim)
        mesh, _ = _createMFEMMesh(file)
        return mesh

    def generateSimpleNGSMesh(self, dim):
        from netgen.read_gmsh import ReadGmsh
        from ngsolve import Mesh

        file = self.generateSimpleGeometry(dim)
        gmesh = ReadGmsh(file)
        return Mesh(gmesh)
        