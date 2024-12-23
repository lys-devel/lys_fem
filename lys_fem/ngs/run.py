import time
import ngsolve
from . import mpi, util
from .mesh import generateMesh
from .material import generateMaterial
from .models import generateModel
from .solver import generateSolver


def run(fem, run=True, save=True, output=True, nthreads=16):
    first = time.time()
    initialize(fem, save, output, nthreads)
    solvers = createSolver(fem)
 
    if run:
        for i, s in enumerate(solvers):
            mpi.print_("------------Solver " + str(i+1) + ": " + s.name + " started --------------------")
            start = time.time()
            if i > 0:
                s.solution.update(solvers[i].solution[0])
            with ngsolve.TaskManager():
                s.execute()
                if False:
                    meshfile = s.exportRefinedMesh()
                    fem.mesher.file=meshfile
                    solvers = createSolver(fem, load=True, print=False)
                    #solvers[i].load(int(util.stepn.get()+1))
                    solvers[i].execute()
            mpi.print_("Calc. time for Solver", str(i+1), ":{:.2f}".format(time.time()-start), " seconds")
            mpi.print_()
    else:
        return mesh, mats, model, solvers
    mpi.print_("Total calculation time: {:.2f}".format(time.time()-first), " seconds")
    mpi.wait()


def initialize(fem, save, output, nthreads):
    mpi.print_("\n----------------------------NGS started ---------------------------")
    mpi.print_()
    mpi.info()

    if save:
        d = fem.saveAsDictionary(parallel=mpi.isParallel())
        with open("input.dic", "w") as f:
            f.write(str(d))

    if output:
        mpi.setOutputFile("output")

    ngsolve.SetNumThreads(nthreads)
    mpi.print_("Number of threads:", nthreads)
    mpi.print_()


def createSolver(fem, load=False, print=True):
    start = time.time()
    mesh = generateMesh(fem)
    mpi.print_("NGS Mesh generated in", '{:.2f}'.format(time.time()-start) ,"seconds : ", mesh.ns[0], "elements, ", mesh.ns[1], "nodes,", len(mesh.GetMaterials()), "domains, ", len(mesh.GetBoundaries()), "boundaries.")
    mpi.print_()

    start = time.time()
    mats = generateMaterial(fem, mesh)
    if print:
        mpi.print_("NGS Variables generated in", '{:.2f}'.format(time.time()-start), "seconds :")
        mpi.print_("\tParameters:", {key: value.shape if len(value.shape)>0 else "scalar" for key, value in mats.items()})
        mpi.print_()

    start = time.time()
    model = generateModel(fem, mesh, mats)
    if print:
        mpi.print_("NGS Models generated in ", '{:.2f}'.format(time.time()-start), "seconds :")
        for m in model.models:
            mpi.print_("\t"+m.name+":", {v.name: v.size for v in m.variables}, "Discretization:", m.discretization)
        mpi.print_()

    start = time.time()
    solvers = generateSolver(fem, mesh, model, load=load)
    if print:
        mpi.print_("NGS Solvers generated in ", '{:.2f}'.format(time.time()-start), "seconds :")
        for i, s in enumerate(solvers):
            mpi.print_("\tSolver", i+1, "(",s.name,"):")
            mpi.print_(s)
        mpi.print_()

    return solvers