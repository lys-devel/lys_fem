import numpy as np
import pyvista as pv
from ..interface import CanvasData3D, VolumeData, SurfaceData

_key_list = {2: pv.CellType.TRIANGLE, 4: pv.CellType.TETRA}
_num_list = {2: 3, 4: 4}


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

    def _rayTrace(self, data, start, end):
        return data.rayTrace(start, end)

    def _remove(self, data):
        data.remove()
