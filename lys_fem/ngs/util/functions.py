import builtins
import numpy as np


def diag(m):
    from .util import NGSFunction
    M = np.zeros(m.shape, dtype=object)
    for i in range(builtins.min(*m.shape)):
        M[i,i] = m[i,i]
    return NGSFunction(M.tolist())


def offdiag(m):
    from .util import NGSFunction
    M = np.zeros(m.shape, dtype=object)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i != j:
                M[i,j] = m[i,j]
    return NGSFunction(M.tolist())


def det(J):
    if J.shape[0] == 3:
        return J[0,0]*J[1,1]*J[2,2] + J[0,1]*J[1,2]*J[2,0] + J[0,2]*J[1,0]*J[2,1] - J[0,2]*J[1,1]*J[2,0] - J[0,1]*J[1,0]*J[2,2] - J[0,0]*J[1,2]*J[2,1]
    elif J.shape[0] == 2:
        return J[0,0]*J[1,1] - J[0,1]*J[1,0]
    elif J.shape[0] == 1:
        return J[0,0]
    
def inv(m):
    if m.shape[0] == 3:
        return NGSFunction([
            [m[1,1]*m[2,2]-m[1,2]*m[2,1], m[0,2]*m[2,1]-m[0,1]*m[2,2], m[0,1]*m[1,2]-m[0,2]*m[1,1]],
            [m[1,2]*m[2,0]-m[1,0]*m[2,2], m[0,0]*m[2,2]-m[0,2]*m[2,0], m[0,2]*m[1,0]-m[0,0]*m[1,2]],
            [m[1,0]*m[2,1]-m[1,1]*m[2,0], m[0,1]*m[2,0]-m[0,0]*m[2,1], m[0,0]*m[1,1]-m[0,1]*m[1,0]]])/m.det()
    elif m.shape[0] == 2:
        return NGSFunction([[m[1,1], -m[0,1]], [-m[1,0], m[0,0]]])/m.det()
    else:
        return 1/m
