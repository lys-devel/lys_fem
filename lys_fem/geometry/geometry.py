
from lys_fem import addGeometry, FEMGeometry
from .geometryGUI import BoxGUI, SphereGUI, RectGUI, LineGUI, DiskGUI, RectFrustumGUI, InfiniteVolumeGUI, QuadGUI, InfinitePlaneGUI


class Box(FEMGeometry):
    def __init__(self, x=0, y=0, z=0, dx=1, dy=1, dz=1):
        self.args = [x, y, z, dx, dy, dz]

    def execute(self, model):
        model.occ.addBox(*self.args)

    @classmethod
    @property
    def type(cls):
        return "box"

    def widget(self):
        return BoxGUI(self)


class Sphere(FEMGeometry):
    def __init__(self, x=0, y=0, z=0, r=1):
        self.args = [x, y, z, r]

    def execute(self, model):
        model.occ.addSphere(*self.args)

    @classmethod
    @property
    def type(cls):
        return "sphere"

    def widget(self):
        return SphereGUI(self)
    

class RectFrustum(FEMGeometry):
    def __init__(self, v1=None, v2=None, v3=None, v4=None, v5=None, v6=None, v7=None, v8=None):
        if v1 is None:
            v1 = [0,0,0]
        if v2 is None:
            v2 = [1,0,0]
        if v3 is None:
            v3 = [1,1,0]
        if v4 is None:
            v4 = [0,1,0]
        if v5 is None:
            v5 = [0,0,1]
        if v6 is None:
            v6 = [1,0,1]
        if v7 is None:
            v7 = [1,1,1]
        if v8 is None:
            v8 = [0,1,1]
        self.args = [v1, v2, v3, v4, v5, v6, v7, v8]

    def execute(self, model):
        pts = [model.occ.addPoint(*v) for v in self.args]

        lines = [model.occ.addLine(pts[0], pts[1]), model.occ.addLine(pts[1], pts[2]), model.occ.addLine(pts[2], pts[3]), model.occ.addLine(pts[3], pts[0])]
        lines.extend([model.occ.addLine(pts[4], pts[5]), model.occ.addLine(pts[5], pts[6]), model.occ.addLine(pts[6], pts[7]), model.occ.addLine(pts[7], pts[4])])      
        lines.extend([model.occ.addLine(pts[0], pts[4]), model.occ.addLine(pts[1], pts[5]), model.occ.addLine(pts[2], pts[6]), model.occ.addLine(pts[3], pts[7])])  

        s1 = model.occ.addSurfaceFilling(model.occ.addCurveLoop([lines[0], lines[1], lines[2], lines[3]]))
        s2 = model.occ.addSurfaceFilling(model.occ.addCurveLoop([lines[4], lines[5], lines[6], lines[7]]))
        s3 = model.occ.addSurfaceFilling(model.occ.addCurveLoop([lines[0], lines[8], lines[4], lines[9]]))
        s4 = model.occ.addSurfaceFilling(model.occ.addCurveLoop([lines[1], lines[9], lines[5], lines[10]]))
        s5 = model.occ.addSurfaceFilling(model.occ.addCurveLoop([lines[2], lines[10], lines[6], lines[11]]))
        s6 = model.occ.addSurfaceFilling(model.occ.addCurveLoop([lines[3], lines[11], lines[7], lines[8]]))
        
        model.occ.addVolume([model.occ.addSurfaceLoop([s1, s3, s4, s5, s6, s2])])

    @classmethod
    @property
    def type(cls):
        return "rectngular frustum"

    def widget(self):
        return RectFrustumGUI(self)
    

class InfiniteVolume(FEMGeometry):
    def __init__(self, a=1, b=1, c=1, A=2, B=2, C=2):
        self.args = [a,b,c,A,B,C]

    def execute(self, model):
        a,b,c,A,B,C = self.args
        RectFrustum((a,b,c), (-a,b,c), (-a,-b,c),(a,-b,c),(A,B,C), (-A,B,C), (-A,-B,C),(A,-B,C)).execute(model)
        RectFrustum((a,b,-c), (-a,b,-c), (-a,-b,-c),(a,-b,-c),(A,B,-C), (-A,B,-C), (-A,-B,-C),(A,-B,-C)).execute(model)
        RectFrustum((a,b,c), (-a,b,c), (-a,b,-c),(a,b,-c),(A,B,C), (-A,B,C), (-A,B,-C),(A,B,-C)).execute(model)
        RectFrustum((a,-b,c), (-a,-b,c), (-a,-b,-c),(a,-b,-c),(A,-B,C), (-A,-B,C), (-A,-B,-C),(A,-B,-C)).execute(model)
        RectFrustum((a,b,c), (a,-b,c), (a,-b,-c),(a,b,-c),(A,B,C), (A,-B,C), (A,-B,-C),(A,B,-C)).execute(model)
        RectFrustum((-a,b,c), (-a,-b,c), (-a,-b,-c),(-a,b,-c),(-A,B,C), (-A,-B,C), (-A,-B,-C),(-A,B,-C)).execute(model)

    @classmethod
    @property
    def type(cls):
        return "infinite volume"

    def widget(self):
        return InfiniteVolumeGUI(self)


class Rect(FEMGeometry):
    def __init__(self, x=0, y=0, z=0, dx=1, dy=1):
        self.args = [x, y, z, dx, dy]

    def execute(self, model):
        model.occ.addRectangle(*self.args)

    @classmethod
    @property
    def type(cls):
        return "rectangle"

    def widget(self):
        return RectGUI(self)


class Disk(FEMGeometry):
    def __init__(self, x=0, y=0, z=0, rx=1, ry=1):
        self.args = [x, y, z, rx, ry]

    def execute(self, model):
        model.occ.addDisk(*self.args)

    @classmethod
    @property
    def type(cls):
        return "disk"

    def widget(self):
        return DiskGUI(self)


class Quad(FEMGeometry):
    def __init__(self, v1=(0,0,0), v2=(1,0,0), v3=(1,1,0), v4=(0,1,0)):
        self.args = [v1, v2 ,v3, v4]

    def execute(self, model):
        pts = [model.occ.addPoint(*v) for v in self.args]
        lines = [model.occ.addLine(pts[0], pts[1]), model.occ.addLine(pts[1], pts[2]), model.occ.addLine(pts[2], pts[3]), model.occ.addLine(pts[3], pts[0])]
        model.occ.addSurfaceFilling(model.occ.addCurveLoop([lines[0], lines[1], lines[2], lines[3]]))

    @classmethod
    @property
    def type(cls):
        return "quad"

    def widget(self):
        return QuadGUI(self)


class InfinitePlane(FEMGeometry):
    def __init__(self, a=1, b=1, A=2, B=2):
        self.args = [a,b,A,B]

    def execute(self, model):
        a,b,A,B = self.args
        Quad((a,b,0), (-a,b,0), (-A,B,0),(A,B,0)).execute(model)
        Quad((a,-b,0), (-a,-b,0), (-A,-B,0),(A,-B,0)).execute(model)
        Quad((a,b,0), (a,-b,0), (A,-B,0),(A,B,0)).execute(model)
        Quad((-a,b,0), (-a,-b,0), (-A,-B,0),(-A,B,0)).execute(model)

    @classmethod
    @property
    def type(cls):
        return "infinite plane"

    def widget(self):
        return InfinitePlaneGUI(self)
    

class Line(FEMGeometry):
    def __init__(self, x1=0, y1=0, z1=0, x2=1, y2=0, z2=0):
        self.args = [x1, y1, z1, x2, y2, z2]

    def execute(self, model):
        p1t = model.occ.addPoint(*self.args[:3])
        p2t = model.occ.addPoint(*self.args[3:])
        model.occ.addLine(p1t, p2t)

    @classmethod
    @property
    def type(cls):
        return "line"

    def widget(self):
        return LineGUI(self)


addGeometry("Add 3D", Box)
addGeometry("Add 3D", Sphere)
addGeometry("Add 3D", RectFrustum)
addGeometry("Add 3D", InfiniteVolume)
addGeometry("Add 2D", Rect)
addGeometry("Add 2D", Disk)
addGeometry("Add 2D", Quad)
addGeometry("Add 2D", InfinitePlane)
addGeometry("Add 1D", Line)
