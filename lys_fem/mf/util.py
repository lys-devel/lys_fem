from . import mfem
from .coef import generateCoefficient

def initialValue(space, x):
    xv = mfem.Vector()
    x_gf = mfem.GridFunction(space)
    x_gf.ProjectCoefficient(x)
    x_gf.GetTrueDofs(xv)
    return xv


def bilinearForm(space, ess_tdof=None, domainInteg=None, boundaryInteg=None):
    # initialization
    if domainInteg is None:
        domainInteg = []
    if not hasattr(domainInteg, "__iter__"):
        domainInteg = [domainInteg]
    if boundaryInteg is None:
        boundaryInteg = []
    if not hasattr(boundaryInteg, "__iter__"):
        boundaryInteg = [boundaryInteg]
    if ess_tdof is None:
        ess_tdof = mfem.intArray()

    # create Bilinear form of mfem
    m = mfem.BilinearForm(space)
    for i in domainInteg:
        m.AddDomainIntegrator(i)
    for i in boundaryInteg:
        m.AddBoundaryIntegrator(i)
    m.Assemble()

    # set it to matrix
    result = mfem.SparseMatrix()
    m.FormSystemMatrix(ess_tdof, result)
    result._bilin = m
    return result

def mixedBilinearForm(space1, space2, ess_tdof1=None, ess_tdof2=None, domainInteg=None, boundaryInteg=None):
    # initialization
    if domainInteg is None:
        domainInteg = []
    if not hasattr(domainInteg, "__iter__"):
        domainInteg = [domainInteg]
    if boundaryInteg is None:
        boundaryInteg = []
    if not hasattr(boundaryInteg, "__iter__"):
        boundaryInteg = [boundaryInteg]
    if ess_tdof1 is None:
        ess_tdof1 = mfem.intArray()
    if ess_tdof2 is None:
        ess_tdof2 = mfem.intArray()

    # create Bilinear form of mfem
    m = mfem.MixedBilinearForm(space1, space2)
    for i in domainInteg:
        m.AddDomainIntegrator(i)
    for i in boundaryInteg:
        m.AddBoundaryIntegrator(i)
    m.Assemble()

    # set it to matrix
    result = mfem.OperatorHandle()
    m.FormRectangularSystemMatrix(ess_tdof1, ess_tdof2, result)
    result._bilin = m
    return result

def linearForm(space, ess_tdof=None, K=None, x0=None, domainInteg=None, boundaryInteg=None):
    # initialization
    if domainInteg is None:
        domainInteg = []
    if not hasattr(domainInteg, "__iter__"):
        domainInteg = [domainInteg]
    if boundaryInteg is None:
        boundaryInteg = []
    if not hasattr(boundaryInteg, "__iter__"):
        boundaryInteg = [boundaryInteg]

    # create Linear form of mfem
    b = mfem.LinearForm(space)
    for i in domainInteg:
        b.AddDomainIntegrator(i)
    for i in boundaryInteg:
        b.AddBoundaryIntegrator(i)
    b.Assemble()

    # set it to vector
    rhs = mfem.Vector()
    mfem.GridFunction(space, b).GetTrueDofs(rhs)
    if ess_tdof is not None and K is not None and x0 is not None:
        K._bilin.EliminateVDofsInRHS(ess_tdof, x0, rhs)
    rhs._lin = b
    return rhs
    
def coefFromVector(space, vec):
    gf = mfem.GridFunction(space)
    gf.SetFromTrueDofs(vec)
    if space.GetVDim() == 1:
        c = mfem.GridFunctionCoefficient(gf)
    else:
        c = mfem.VectorGridFunctionCoefficient(gf)
    return c

def projectVectorCoef(cv):
    cx = mfem.InnerProductCoefficient(cv, mfem.VectorConstantCoefficient([1,0,0]))
    cy = mfem.InnerProductCoefficient(cv, mfem.VectorConstantCoefficient([0,1,0]))
    cz = mfem.InnerProductCoefficient(cv, mfem.VectorConstantCoefficient([0,0,1]))
    return cx, cy, cz

def generateDomainCoefficient(mesh, conditions):
    coefs = {}
    for c in conditions:
        for d in mesh.attributes:
            if c.domains.check(d):
                coefs[d] = c.values
    return generateCoefficient(coefs, mesh.Dimension())


def generateSurfaceCoefficient(mesh, conditions):
    bdr_stress = {}
    for b in conditions:
        for d in mesh.bdr_attributes:
            if b.boundaries.check(d):
                bdr_stress[d] = b.values
    return generateCoefficient(bdr_stress, mesh.Dimension())