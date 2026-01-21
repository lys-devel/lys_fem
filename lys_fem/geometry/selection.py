
class GeometrySelection:
    def __init__(self, geometryType="Domain", selection=None):
        if isinstance(selection, GeometrySelection):
            geometryType = selection.geometryType
            selection = selection.getSelection()
        if selection is None:
            selection = []
        self._geom = geometryType
        if selection == "all":
            self._selection = "all"
        elif isinstance(selection, str):
            self._selection = [selection]
        else:
            self._selection = list(selection)
        
    def __getitem__(self, index):
        return self._selection[index]

    def getSelection(self, geom=None):
        if geom is None:
            return self._selection

    def setSelection(self, value):
        self._selection = value

    def append(self, value):
        self._selection.append(value)
        self._selection = sorted(self._selection)

    def remove(self, value):
        self._selection.remove(value)

    def clear(self):
        self._selection.clear()

    def selectionType(self):
        if self._selection == "all":
            return "All"
        else:
            if len(self._selection) == 0:
                return "Group"
            elif isinstance(self._selection[0], str):
                return "Group"
            else:
                return "Selected"

    def get(self, geom):
        """
        Get selected geometries as a list of integers.

        Args:
            geom(GmshGeometry): The geometry object.
        """
        if self._geom == "Domain":
            attrs = geom.geometryAttributes(geom.dimension)
        elif self._geom == "Boundary":
            attrs = geom.geometryAttributes(geom.dimension-1)
        elif self._geom == "Volume":
            attrs = geom.geometryAttributes(3)
        elif self._geom == "Surface":
            attrs = geom.geometryAttributes(2)
        elif self._geom == "Edge":
            attrs = geom.geometryAttributes(1)
        elif self._geom == "Point":
            attrs = geom.geometryAttributes(0)
        return [attr for attr in attrs if self._check(geom, attr)]

    def _check(self, geom, item):
        if self._selection == "all":
            return True
        if len(self._selection) == 0:
            return False
        if isinstance(self._selection[0], str):
            return any([item in geom.groups[s].get(geom) for s in self._selection])
        else:
            return item in self._selection

    @property
    def geometryType(self):
        return self._geom

    def saveAsDictionary(self):
        return {"selection": self._selection, "geometryType": self._geom}

    @staticmethod
    def loadFromDictionary(d):
        return GeometrySelection(d["geometryType"], d["selection"])

