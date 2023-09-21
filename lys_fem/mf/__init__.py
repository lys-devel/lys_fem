from .mesh import generateMesh
from .material import generateMaterial
from .models import generateModel


def run_mfem(file):
    from ..fem import FEMProject
    fem = FEMProject(2)
    fem.loadFromDictionary(file)
    run(fem)


def run(fem):
    geom = fem.geometryGenerator.generateGeometry(fem.dimension)
    fem.mesher.export(geom, "mesh.msh")
    mesh = generateMesh("mesh.msh")
    material = generateMaterial(fem, geom)
    model = generateModel(fem, geom, mesh, material)
    print(material)
    print(mesh)
    print(model)
    x = model.solve()
    return
    t, dt = 0.0, 0.01
    for ti in range(51):
        t, dt = model.step(t, dt)
        print(ti, t)
        # oper.SetParameters(u)
