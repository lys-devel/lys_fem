import gmsh
import numpy as np
from lys_fem import mf

if mf.parallel:
    import mfem.par as mfem
    from mpi4py import MPI
else:
    import mfem.ser as mfem


def generateMesh(file):
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
    if mf.parallel:
        mesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    return mesh
