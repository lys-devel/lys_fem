import time
from .mpi import print_, info, isParallel, wait
from .mesh import generateMesh
from .material import generateMaterial
from .models import generateModel
from .solver import generateSolver


def run(fem, run=True, save=True):
    print_("\n----------------------------NGS started ---------------------------")
    print_()
    info()

    if save:
        d = fem.saveAsDictionary(parallel=isParallel())
        with open("input.dic", "w") as f:
            f.write(str(d))

    start = time.time()
    mesh = generateMesh(fem)
    print_("NGS Mesh generated in", '{:.2f}'.format(time.time()-start) ,"seconds : ", mesh.ne, "elements, ", mesh.nv, "nodes,", len(mesh.GetMaterials()), "domains, ", len(mesh.GetBoundaries()), "boundaries.")
    print_()

    start = time.time()
    mats = generateMaterial(fem, mesh)
    print_("NGS Variables generated in", '{:.2f}'.format(time.time()-start), "seconds :")
    print_("\tParameters:", {key: value.shape if len(value.shape)>0 else "scalar" for key, value in mats.items()})
    print_()

    start = time.time()
    model = generateModel(fem, mesh, mats)
    print_("NGS Models generated in ", '{:.2f}'.format(time.time()-start), "seconds :")
    for m in model.models:
        print_("\t"+m.name+":", {v.name: v.size for v in m.variables})
    print_()

    start = time.time()
    solvers = generateSolver(fem, mesh, model)
    print_("NGS Solvers generated in ", '{:.2f}'.format(time.time()-start), "seconds :")
    for i, s in enumerate(solvers):
        print_("\tSolver", i, ":", s.name)
        for j, step in enumerate(s.obj.steps):
            print_("\tStep", j, ":", step)
    print_()
 
    if run:
        for i, s in enumerate(solvers):
            print_("------------Solver " + str(i) + ": " + s.name + " started --------------------")
            start = time.time()
            if i > 0:
                s.solution.update(solvers[i].solution[0])
            s.execute()
            print_()
            print_("Total calculation time: ", time.time()-start, " seconds")
            print_()
    else:
        return mesh, mats, model, solvers
    wait()
