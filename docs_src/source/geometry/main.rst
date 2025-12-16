geometry module
========================

lys_fem.geometry is a gmsh wrapper that provides simplified interface.

The simplest example of this module::

    from lys_fem.geometry import Box, GmshGeometry, GmshMesh

    # Generate simple geometry with two boxes
    b1 = Box(x=0, y=0, z=0, dx=1, dy=1, dz=1)
    b2 = Box(x=1, y=0, z=0, dx=1, dy=1, dz=1)
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
