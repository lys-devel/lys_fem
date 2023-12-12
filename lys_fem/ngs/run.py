from .mesh import generateMesh
#from .material import generateMaterial
from .models import generateModel
from .solver import generateSolver


def run(fem, run=True):
    mesh = generateMesh(fem)
    print(mesh)
    models = generateModel(fem, mesh, None)
    print(models)
    solvers = generateSolver(fem, mesh, models)
    print(solvers)
 
    if run:
        for i, s in enumerate(solvers):
            print("------------Solver " + str(i) + ": " + s.name + " started --------------------")
            import time
            start = time.time()
            s.execute()
            print(time.time()-start)
    else:
        return mesh, material, models, solvers

