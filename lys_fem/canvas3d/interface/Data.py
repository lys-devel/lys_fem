import warnings
import io
import numpy as np

from lys import Wave, filters
from lys.Qt import QtCore
from lys.errors import NotImplementedWarning

from .CanvasBase import CanvasPart3D, saveCanvas
from .WaveData import WaveData3D
from .Volume import VolumeData
from .Surface import SurfaceData


class CanvasData3D(CanvasPart3D):
    dataChanged = QtCore.pyqtSignal()
    """pyqtSignal that is emittd when data is added/removed/changed."""

    def __init__(self, canvas):
        super().__init__(canvas)
        self._Datalist = []
        canvas.saveCanvas.connect(self._save)
        canvas.loadCanvas.connect(self._load)

    @saveCanvas
    def append(self, wave, appearance={}, offset=(0, 0, 0, 0), filter=None):
        """
        Append Wave to canvas.

        Args:
            wave(Wave): The data to be added.
            appearance(dict): The dictionary that determins appearance. See :meth:`.WaveData.saveAppearance` for detail.
            offset(tuple  of length 4): See :meth:`.WaveData.setOffset`
            filter(filter): See :meth:`.WaveData.setFilter`
        """
        func = {"volume": self._appendVolume, "surface": self._appendSurface}
        # When multiple data is passed
        if isinstance(wave, list) or isinstance(wave, tuple):
            return [self.append(ww) for ww in wave]
        # Update
        if isinstance(wave, WaveData3D):
            return self.append(wave.getWave(),
                               axis=wave.getAxis(),
                               appearance=wave.saveAppearance(),
                               offset=wave.getOffset(),
                               filter=wave.getFilter())
        type = self.__checkType(wave)
        obj = func[type](wave)
        obj.setOffset(offset)
        obj.setFilter(filter)
        obj.loadAppearance(appearance)
        obj.modified.connect(self.dataChanged)
        self._Datalist.append(obj)
        self.dataChanged.emit()
        return obj

    def __checkType(self, wav):
        return "surface"
        raise RuntimeError("[Graph] Can't append this data. shape = " + str(wav.data.shape))

    @ saveCanvas
    def remove(self, obj):
        """
        Remove data from canvas.

        Args:
            obj(WaveData): WaveData object to be removed.
        """
        if hasattr(obj, '__iter__'):
            for o in obj:
                self.remove(o)
            return
        self._remove(obj)
        self._Datalist.remove(obj)
        obj.modified.disconnect(self.dataChanged)
        self.dataChanged.emit()

    @ saveCanvas
    def clear(self):
        """
        Remove all data from canvas.
        """
        while len(self._Datalist) != 0:
            self.remove(self._Datalist[0])

    def getWaveData(self, type="all"):
        """
        Return list of WaveData object that is specified by *type*.

        Args:
            type('all', 'line', 'image', 'rgb', 'vector', or 'contour'): The data type to be returned.
        """
        if type == "all":
            return self._Datalist
        elif type == "volume":
            return [data for data in self._Datalist if isinstance(data, VolumeData)]
        elif type == "surface":
            return [data for data in self._Datalist if isinstance(data, SurfaceData)]

    def getVolume(self):
        """
        Return all VolumeData in the canvas.
        """
        return self.getWaveData("volume")

    def getSurface(self):
        """
        Return all SurfaceData in the canvas.
        """
        return self.getWaveData("surface")

    def rayTrace(self, start, end, type="all"):
        data = self.getWaveData(type)
        distance = np.linalg.norm(end - start)
        for d in data:
            point = self._rayTrace(d, start, end)
            if len(point) == 0:
                continue
            if np.linalg.norm(point - start) < distance:
                distance = np.linalg.norm(point - start)
                res = d
        return res

    def _save(self, dictionary):
        dic = {}
        for i, data in enumerate(self._Datalist):
            dic[i] = {}
            dic[i]['File'] = None
            b = io.BytesIO()
            data.getWave().export(b)
            dic[i]['Wave_npz'] = b.getvalue()
            dic[i]['Axis'] = data.getAxis()
            dic[i]['Appearance'] = str(data.saveAppearance())
            dic[i]['Offset'] = str(data.getOffset())
            dic[i]['ZOrder'] = data.getZOrder()
            if data.getFilter() is None:
                dic[i]['Filter'] = None
            else:
                dic[i]['Filter'] = filters.toString(data.getFilter())
        dictionary['Datalist'] = dic

    def _load(self, dictionary):
        if 'Datalist' in dictionary:
            self.Clear()
            dic = dictionary['Datalist']
            i = 0
            while i in dic:
                w = Wave(io.BytesIO(dic[i]['Wave_npz']))
                obj = self.Append(w)
                self.__loadMetaData(obj, dic[i])
                i += 1

    def __loadMetaData(self, obj, d):
        if 'Offset' in d:
            obj.setOffset(eval(d['Offset']))
        filter = d.get('Filter', None)
        if filter is not None:
            obj.setFilter(filters.fromString(filter))
        if 'Appearance' in d:
            obj.loadAppearance(eval(d['Appearance']))

    def _remove(self, data):
        raise NotImplementedError(str(type(self)) + " does not implement _remove(data) method.")

    def _appendVolume(self, wave):
        warnings.warn(str(type(self)) + " does not implement _append1d(wave, axis) method.", NotImplementedWarning)

    def _appendSurface(self, wave):
        warnings.warn(str(type(self)) + " does not implement _append2d(wave, axis) method.", NotImplementedWarning)

    def _rayTrace(self, data, start, end):
        warnings.warn(str(type(self)) + " does not implement _rayTrace(data) method.", NotImplementedWarning)
