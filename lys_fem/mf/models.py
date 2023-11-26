from . import mfem
from ..fem import DirichletBoundary

modelList = {}


def addMFEMModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return [modelList[m.name](m, mesh, mat) for m in fem.models]


class MFEMModel:
    def __init__(self, model):
        self._model = model
        self.__x0 = None
        self.__K = None
        self.__M = None
        self.__b = None
        self.__dK = None
        self.__dM = None
        self.__db = None

    def essential_tdof_list(self, space):
        res = []
        for axis, b in self.dirichletCondition.items():
            if len(b) == 0:
                continue
            ess_bdr = mfem.intArray(space.GetMesh().bdr_attributes.Max())
            ess_bdr.Assign(0)
            for i in b:
                ess_bdr[i - 1] = 1
            ess_tdof_list = mfem.intArray()
            space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list, axis)
            res.extend([i for i in ess_tdof_list])
        return mfem.intArray(res)


    @property
    def dirichletCondition(self):
        conditions = [b for b in self._model.boundaryConditions if isinstance(b, DirichletBoundary)]
        bdr_dir = {i: [] for i in range(self._model.variableDimension())}
        for b in conditions:
            for axis, check in enumerate(b.components):
                if check:
                    bdr_dir[axis].extend(b.boundaries.getSelection())
        return bdr_dir

    @property
    def variableName(self):
        return self._model.variableName

    @property
    def preconditioner(self):
        return None
    
    @property
    def timeUnit(self):
        return 1
    
    @property
    def x0(self):
        return self.__x0
    
    @x0.setter
    def x0(self, value):
        self.__x0 = value

    @property
    def K(self):
        return self.__K
    
    @K.setter
    def K(self, value):
        self.__K = value

    @property
    def M(self):
        return self.__M
    
    @M.setter
    def M(self, value):
        self.__M = value
        
    @property
    def b(self):
        return self.__b
    
    @b.setter
    def b(self, value):
        self.__b = value

    @property
    def grad_Kx(self):
        return self.__dK
    
    @grad_Kx.setter
    def grad_Kx(self, value):
        self.__dK = value

    @property
    def grad_Mx(self):
        return self.__dM
    
    @grad_Mx.setter
    def grad_Mx(self, value):
        self.__dM = value
        
    @property
    def grad_b(self):
        return self.__db
    
    @grad_b.setter
    def grad_b(self, value):
        self.__db = value

class MFEMLinearModel(MFEMModel):
    def update(self, x):
        pass

    @property
    def grad_Mx(self):
        return self.M

    @property
    def grad_Kx(self):
        return self.K


class MFEMNonlinearModel(MFEMModel):
    pass
