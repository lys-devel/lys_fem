import datetime

from mpi4py import MPI
from .mesh import generateMesh
from .material import generateMaterial
from .models import generateModel
from .solver import LinearSolver


def print_(*args):
    if MPI.COMM_WORLD.rank == 0:
        print(*args)


def run_mfem(file):
    from ..fem import FEMProject
    fem = FEMProject(2)
    fem.loadFromDictionary(file)
    run(fem)


def run(fem):
    print_("-----------------------------------------------------------")
    print_("lys_fem starts at", datetime.datetime.now(), " with ", str(MPI.COMM_WORLD.size), "processors")
    geom = fem.geometryGenerator.generateGeometry(fem.dimension)
    fem.mesher.export(geom, "mesh.msh")
    mesh = generateMesh("mesh.msh")
    print_("Mesh generated: ", str(len([1 for _ in mesh.attributes])), "domains,", str(len([1 for _ in mesh.bdr_attributes])), "boundaries")
    material = generateMaterial(fem, geom)
    print_("Material generated for ", str([name for name in material.keys()]))
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
