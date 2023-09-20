import numpy as np
import pyvista as pv
from lys.Qt import QtGui

from ..interface import CanvasData3D, VolumeData, SurfaceData

_key_list = {"triangle": pv.CellType.TRIANGLE, "tetra": pv.CellType.TETRA, "hexa": pv.CellType.HEXAHEDRON, "quad": pv.CellType.QUAD, "pyramid": pv.CellType.PYRAMID, "prism": pv.CellType.WEDGE}
_num_list = {"line": 2, "triangle": 3, "tetra": 4, "hexa": 8, "quad": 4, "pyramid": 5, "prism": 6}


class _pyvistaVolume(VolumeData):
    """Implementation of VolumeData for pyvista"""

    def __init__(self, canvas, wave):
        super().__init__(canvas, wave)
        mesh = pv.UnstructuredGrid({_key_list[key]: item for key, item in wave.note["elements"].items()}, wave.x)
        self._obj = canvas.plotter.add_mesh(mesh, show_edges=True)

    def remove(self):
        self.canvas().plotter.remove_actor(self._obj)

    def _updateData(self):
        return

    def _setVisible(self, visible):
        pass

    def _setColor(self, color):
        pass


class _pyvistaSurface(SurfaceData):
    """Implementation of LineData for matplotlib"""

    def __init__(self, canvas, wave):
        super().__init__(canvas, wave)
        faces = [np.ravel(np.hstack([np.ones((len(faces), 1), dtype=int) * _num_list[key], faces])) for key, faces in wave.note["elements"].items()]
        self._mesh = pv.PolyData(wave.x, np.hstack(faces))
        self._edges = self._mesh.extract_feature_edges(boundary_edges=True)
        self._obj = canvas.plotter.add_mesh(self._mesh)
        self._obje = canvas.plotter.add_mesh(self._edges, color='k', line_width=3)

    def remove(self):
        self.canvas().plotter.remove_actor(self._obj)
        self.canvas().plotter.remove_actor(self._obje)

    def rayTrace(self, start, end):
        return self._mesh.ray_trace(start, end, first_point=True)[0]

    def _setColor(self, color, type):
        if type == "color":
            c = QtGui.QColor(color)
            self._obj.GetProperty().SetColor([c.redF(), c.greenF(), c.blueF()])

    def _showEdges(self, b):
        self._obj.GetProperty().show_edges = b

    def _updateData(self):
        return

    def _setVisible(self, visible):
        pass


class _pyvistaLine(SurfaceData):
    """Implementation of LineData for matplotlib"""

    def __init__(self, canvas, wave):
        super().__init__(canvas, wave)
        faces = [np.ravel(np.hstack([np.ones((len(faces), 1), dtype=int) * _num_list[key], faces])) for key, faces in wave.note["elements"].items()]
        self._mesh = pv.PolyData(wave.x, lines=np.hstack(faces))
        self._obj = canvas.plotter.add_mesh(self._mesh, show_edges=True)

    def remove(self):
        self.canvas().plotter.remove_actor(self._obj)

    def rayTrace(self, start, end):
        return self._mesh.ray_trace(start, end, first_point=True)[0]

    def _updateData(self):
        return

    def _setVisible(self, visible):
        pass


class _pyvistaData(CanvasData3D):
    def _appendVolume(self, wave):
        return _pyvistaVolume(self.canvas(), wave)

    def _appendSurface(self, wave):
        return _pyvistaSurface(self.canvas(), wave)

    def _appendLine(self, wave):
        return _pyvistaLine(self.canvas(), wave)

    def _rayTrace(self, data, start, end):
        return data.rayTrace(start, end)

    def _remove(self, data):
        data.remove()
