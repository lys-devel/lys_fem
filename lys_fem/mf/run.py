
import datetime
from lys_fem import mf

if mf.parallel:
    from mpi4py import MPI


def run(fem):
    #from .solver import LinearSolver
    #from .models import generateModel
    from .mesh import generateMesh
    from .material import generateMaterial

    print_("-----------------------------------------------------------")
    if mf.parallel:
        print_("lys_fem starts at", datetime.datetime.now(), " with ", str(MPI.COMM_WORLD.size), "processors")
    else:
        print_("lys_fem starts at", datetime.datetime.now(), "in serial mode")
    geom = fem.geometryGenerator.generateGeometry(fem.dimension)
    fem.mesher.export(geom, "mesh.msh")
    mesh = generateMesh("mesh.msh")
    print_("Mesh generated: ", str(len([1 for _ in mesh.attributes])), "domains,", str(len([1 for _ in mesh.bdr_attributes])), "boundaries")
    material = generateMaterial(fem, geom)
    print_("Material generated for ", str([name for name in material.keys()]))
    return
    model = generateModel(fem, geom, mesh, material)
    print_("Model generated:")
    solver = LinearSolver(model)
    x = solver.solve()
    return
    t, dt = 0.0, 0.01
    for ti in range(51):
        t, dt = model.step(t, dt)
        print(ti, t)
        # oper.SetParameters(u)


def print_(*args):
    if mf.parallel:
        if MPI.COMM_WORLD.rank != 0:
            return
    print(*args)
