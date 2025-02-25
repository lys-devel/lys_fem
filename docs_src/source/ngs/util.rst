util module
---------------

lys_fem.ngs.util is a partial NGSolve wrapper that provides simplified interface for developers.

The simplest example of this module::

    from lys_fem.ngs.util import H1, FiniteElementSpace, grad, dx, Solver

    # H1 space and weakform of Poisson equation.
    fs = H1("H1")
    u, v = fs.trial, fs.test
    wf = grad(u).dot(grad(v))*dx

    # Make finite element space with a mesh, and make grid function on it.
    fes = FiniteElementSpace([fs], m)
    g = fes.gridFunction()

    # Solve Poisson equation on given finite element space.
    Solver(fes, wf, linear={"solver": "pardiso"}).solve(g)
