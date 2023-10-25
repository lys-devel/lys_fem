import gmsh
import numpy as np
from lys_fem import mf

if mf.parallel:
    import mfem.par as mfem
    from mpi4py import MPI
else:
    import mfem.ser as mfem


def generateMesh(fem, geom, file="mesh.msh"):
    if mf.parallel:
        if MPI.COMM_WORLD.rank == 0:
            fem.mesher.export(geom, file)
        MPI.COMM_WORLD.scatter([0] * MPI.COMM_WORLD.size, root=0)
    else:
        fem.mesher.export(geom, file)
    mesh = mfem.Mesh(file, 1, 1)
    if len([i for i in mesh.bdr_attributes]) == 0:  # For 1D mesh, we have to set boundary manually.
        # Load file by gmsh
        model = gmsh.model()
        model.add("Default")
        model.setCurrent("Default")
        gmsh.merge(file)

        # Get all boundary nodes
        s = set(np.array([model.mesh.getNodes(*obj, includeBoundary=True)[0][:2] for obj in model.getEntities(1)]).flatten())

        # Set the boundary nodes to mesh object.
        for v in s:
            mesh.AddBdrPoint(v, v)
        mesh.SetAttributes()
    nv = mesh.GetNV()
    if mf.parallel:
        mesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    return mesh, nv
