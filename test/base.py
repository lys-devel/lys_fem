import unittest
import os
import shutil

from lys_fem import mf


class FEMTestCase(unittest.TestCase):
    path = "test/run"

    def setUp(self):
        os.makedirs(self.path, exist_ok=True)
        self._cwd = os.getcwd()
        os.chdir(self.path)

    def tearDown(self):
        os.chdir(self._cwd)
        if mf.mfem.isRoot:
            shutil.rmtree(self.path)
