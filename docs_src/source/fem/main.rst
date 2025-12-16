fem module
========================

lys_fem.fem module is a high-level interface for ngsolve and gmsh, which hinders low-level weakforms, time discretization, etc.

The simplest example of this module::

    from lys_fem import FEMProject, geometries

    # Generate simple geometry with two boxes
    b1 = geometries.Box(x=0, y=0, z=0, dx=1, dy=1, dz=1)
    b2 = geometries.Box(x=1, y=0, z=0, dx=1, dy=1, dz=1)
    geom = GmshGeometry([b1, b2])

    print(geom) # Name: Geometry1,  12 points, 20 edges, 11 surfaces, 2 volumes

    # Generate mesh
    mesh = GmshMesh(geom)
    print(mesh) # Gmsh mesh object, 433 nodes, 2338 elements

    # It is also possible to create a mesh from boxes directry.
    mesh2 = GmshMesh([b1, b2])
    print(mesh2)


.. toctree::
   :maxdepth: 1
   :caption: API References:

   geometry
   mesh
   geometries
