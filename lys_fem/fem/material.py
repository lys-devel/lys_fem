from lys.Qt import QtCore


class MaterialList(QtCore.QObject):
    itemChanged = QtCore.pyqtSignal()

    def __init__(self, materials=[]):
        super().__init__()
        self._list = materials

    def append(self, item):
        self._list.append(item)
        self.itemChanged.emit()

    def remove(self, item):
        self._list.remove(item)
        self.itemChanged.emit()

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __iter__(self):
        return iter(self._list)

    def saveAsDictionary(self):
        return {"materials": [m.saveAsDictionary() for m in self._list]}

    def loadFromDictionary(self, d):
        self._list = [Material.loadFromDictionary(m) for m in d["materials"]]


class Material(QtCore.QObject):
    itemChanged = QtCore.pyqtSignal()

    def __init__(self, name, domains=None, params=None):
        super().__init__()
        self._name = name
        if domains is None:
            domains = []
        self._domains = domains
        if params is None:
            params = []
        self._list = params

    @property
    def name(self):
        return self._name

    def append(self, item):
        self._list.append(item)
        self.itemChanged.emit()

    def remove(self, item):
        self._list.remove(item)
        self.itemChanged.emit()

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __iter__(self):
        return iter(self._list)

    @property
    def domains(self):
        return self._domains

    @domains.setter
    def domains(self, value):
        if value == "all":
            self._domains = "all"
        else:
            self._domains = list(value)

    def saveAsDictionary(self):
        return {"name": self._name, "domains": self.domains, "params": [p.saveAsDictionary() for p in self._list]}

    @staticmethod
    def loadFromDictionary(d):
        params = [FEMParameter.loadFromDictionary(p) for p in d["params"]]
        return Material(d["name"], d["domains"], params)


class FEMParameter:
    def saveAsDictionary(self):
        d = vars(self)
        d["paramsName"] = self.name
        return d

    @staticmethod
    def loadFromDictionary(d):
        cls_list = set(sum(materialParameters.values(), []))
        cls_dict = {value.name: value for value in cls_list}

        d = dict(d)
        cls = cls_dict[d["paramsName"]]
        del d["paramsName"]
        return cls(**d)


class ElasticParameters(FEMParameter):
    def __init__(self, rho, C, type="lame"):
        self.rho = rho
        self.C = C
        self.type = type

    @classmethod
    @property
    def name(cls):
        return "Elasticity"


materialParameters = {"Acoustics": [ElasticParameters]}
