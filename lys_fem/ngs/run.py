import time
from .mesh import generateMesh
from .material import generateMaterial
from .models import generateModel
from .solver import generateSolver


def run(fem, run=True):
    print("\n----------------------------NGS started ---------------------------")
    print()

    d = fem.saveAsDictionary()
    with open("input.dic", "w") as f:
        f.write(str(d))

    start = time.time()
    mesh = generateMesh(fem)
    print("NGS Mesh generated in", '{:.2f}'.format(time.time()-start) ,"second : ", mesh.ne, "elements, ", mesh.nv, "nodes.")
    print("\tDomains:", mesh.GetMaterials())
    print("\tBoundaries:", mesh.GetBoundaries())
    print()

    start = time.time()
    mats = generateMaterial(fem, mesh)
    print("NGS Material generated in", '{:.2f}'.format(time.time()-start), ":")
    print("\tParameters:", {key: value.shape if len(value.shape)>0 else "scalar" for key, value in mats.items()})
    print()

    start = time.time()
    model = generateModel(fem, mesh, mats)
    print("NGS Models generated in ", '{:.2f}'.format(time.time()-start), ":")
    for m in model.models:
        print("\t"+m.name+":", [v.name for v in m.variables])
    print()

    start = time.time()
    solvers = generateSolver(fem, mesh, model)
    print("NGS Solvers generated in ", '{:.2f}'.format(time.time()-start), ":"+str([s.name for s in solvers]))
    print()
 
    if run:
        for i, s in enumerate(solvers):
            print("------------Solver " + str(i) + ": " + s.name + " started --------------------")
            start = time.time()
            s.execute()
            print(time.time()-start)
    else:
        return mesh, mats, model, solvers

