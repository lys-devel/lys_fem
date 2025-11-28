import numpy as np
import ngsolve
from .coef import NGSFunction
from .fields import VolumeField

x = NGSFunction(ngsolve.x, name="x")
y = NGSFunction(ngsolve.y, name="y")
z = NGSFunction(ngsolve.z, name="z")

c = NGSFunction(2.99792458e8, name="c")
e = NGSFunction(-1.602176634e-19, name="e")
pi = NGSFunction(np.pi, name="pi")
k_B = NGSFunction(1.3806488e-23, name="k_B")
g_e = NGSFunction(1.760859770e11, name="g_e")
mu_0 = NGSFunction(1.25663706e-6, name="mu_0")
mu_B = NGSFunction(9.2740100657e-24 , name="mu_B")
eps_0 = NGSFunction(8.8541878128e-12, name="eps_0")

Ve = VolumeField(name = "Ve")

t = NGSFunction(ngsolve.Parameter(0), name="t", tdep=True)
dti = NGSFunction(ngsolve.Parameter(-1), name="dti", tdep=True)
stepn = NGSFunction(ngsolve.Parameter(0), name = "step", tdep=True)


def asdict():
    return {
        "x": x, "y": y, "z": z, "t": t, "dti": dti,
        "c": c, "e": e, "pi": pi, "k_B": k_B, "g_e": g_e, "mu_0": mu_0, "mu_B": mu_B, "eps_0": eps_0, 
        "Ve": Ve
    }