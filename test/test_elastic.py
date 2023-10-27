import unittest
import os
import shutil

from lys_fem import FEMProject


class elasticity_test(unittest.TestCase):
    path = "test/testData"

    def setUp(self):
        os.makedirs(self.path, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_1d(self):
        proj = FEMProject(1)
