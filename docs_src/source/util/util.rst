util module
---------------

lys_fem.util is a NGSolve wrapper that provides simplified interface for developers.

The simplest example of this module::

    from lys_fem.geometry import Box, GmshMesh
    from lys_fem.util import Mesh, H1, FiniteElementSpace, grad, dx, Solver, x

    # Create gmsh mesh, and translate it to ngsolve mesh.
    gm = GmshMesh([Box()])
    m = Mesh(gm)

    # H1 space (variable name "u") and weakform of Poisson equation.
    fs = H1("u")
    u, v = fs.trial, fs.test
    wf = grad(u).dot(grad(v))*dx

    # Make finite element space with a mesh
    fes = FiniteElementSpace(fs, m)

    # initialize grid function g(x,y,z) = x
    g = fes.gridFunction(x)

    # Solve Poisson equation on given finite element space.
    Solver(fes, wf, linear={"solver": "pardiso"}).solve(g)
