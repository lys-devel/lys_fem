import ngsolve
from .util import prod


class MeshFreeSpace:
    def __init__(self, name, size, isScalar):
        self._name = name
        self._size = size
        self._scalar = isScalar

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size
    
    @property
    def isScalar(self):
        return self._scalar and self._size==1


class H1(MeshFreeSpace):
    def __init__(self, name, size, dirichlet=None, isScalar=True, **kwargs):
        super().__init__(name, size, isScalar)
        self._kwargs = dict(kwargs)
        if dirichlet is None:
            dirichlet = [None] * size
        self._dirichlet = dirichlet

    def eval(self, mesh):
        fess = []
        for i in range(self.size):
            if self._dirichlet[i] is None:
                fess.append(ngsolve.H1(mesh, **self._kwargs))
            else:
                fess.append(ngsolve.H1(mesh, dirichlet=self._dirichlet[i], **self._kwargs))
        return prod(fess)


class L2(MeshFreeSpace):
    def __init__(self, name, size, dirichlet=None, isScalar=True, **kwargs):
        super().__init__(name, size, isScalar)
        self._kwargs = dict(kwargs)
        self._dirichlet = dirichlet

    def eval(self, mesh):
        fess = []
        for i in range(self.size):
            if self._dirichlet[i] is None:
                fess.append(ngsolve.L2(mesh, **self._kwargs))
            else:
                fess.append(ngsolve.L2(mesh, dirichlet=self._dirichlet[i], **self._kwargs))
        return prod(fess)
