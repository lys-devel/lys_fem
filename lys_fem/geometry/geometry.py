import numpy as np
import sympy as sp
from lys_fem import addGeometry, FEMGeometry
from lys_fem.fem import FEMCoefficient
from .geometryGUI import BoxGUI, SphereGUI, RectGUI, LineGUI, DiskGUI, RectFrustumGUI, InfiniteVolumeGUI, QuadGUI, InfinitePlaneGUI


class Box(FEMGeometry):
    def __init__(self, x=0, y=0, z=0, dx=1, dy=1, dz=1):
        super().__init__([x, y, z, dx, dy, dz])

    def execute(self, model, trans):
        args = trans(self.args, unit="m")
        model.occ.addBox(*args)

    @classmethod
    @property
    def type(cls):
        return "box"

    def widget(self):
        return BoxGUI(self)


class Sphere(FEMGeometry):
    def __init__(self, x=0, y=0, z=0, r=1):
        super().__init__([x, y, z, r])

    def execute(self, model, trans):
        args = trans(self.args, unit="m")
        model.occ.addSphere(*args, angle2=0, angle3=np.pi/2)
        model.occ.addSphere(*args, angle2=0, angle3=np.pi)
        model.occ.addSphere(*args, angle2=0, angle3=3*np.pi/2)
        model.occ.addSphere(*args, angle2=0, angle3=2*np.pi)
        model.occ.addSphere(*args, angle1=0, angle3=np.pi/2)
        model.occ.addSphere(*args, angle1=0, angle3=np.pi)
        model.occ.addSphere(*args, angle1=0, angle3=3*np.pi/2)
        model.occ.addSphere(*args, angle1=0, angle3=2*np.pi)

    @classmethod
    @property
    def type(cls):
        return "sphere"

    def widget(self):
        return SphereGUI(self)
    

class RectFrustum(FEMGeometry):
    def __init__(self, v1=(0,0,0), v2=(1,0,0), v3=(1,1,0), v4=(0,1,0), v5=(0,0,1), v6=(1,0,1), v7=(1,1,1), v8=(0,1,1)):
        super().__init__([v1, v2, v3, v4, v5, v6, v7, v8])

    def execute(self, model, trans):
        pts = [model.occ.addPoint(*trans(v, unit="m")) for v in self.args]

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
    type = "infinite volume"
    def __init__(self, a=1, b=1, c=1, A=2, B=2, C=2):
        super().__init__([a,b,c,A,B,C])

    def execute(self, model, trans):
        a,b,c,A,B,C = self.args
        RectFrustum((a,b,c), (-a,b,c), (-a,-b,c),(a,-b,c),(A,B,C), (-A,B,C), (-A,-B,C),(A,-B,C)).execute(model, trans)
        RectFrustum((a,b,-c), (-a,b,-c), (-a,-b,-c),(a,-b,-c),(A,B,-C), (-A,B,-C), (-A,-B,-C),(A,-B,-C)).execute(model, trans)
        RectFrustum((a,b,c), (-a,b,c), (-a,b,-c),(a,b,-c),(A,B,C), (-A,B,C), (-A,B,-C),(A,B,-C)).execute(model, trans)
        RectFrustum((a,-b,c), (-a,-b,c), (-a,-b,-c),(a,-b,-c),(A,-B,C), (-A,-B,C), (-A,-B,-C),(A,-B,-C)).execute(model, trans)
        RectFrustum((a,b,c), (a,-b,c), (a,-b,-c),(a,b,-c),(A,B,C), (A,-B,C), (A,-B,-C),(A,B,-C)).execute(model, trans)
        RectFrustum((-a,b,c), (-a,-b,c), (-a,-b,-c),(-a,b,-c),(-A,B,C), (-A,-B,C), (-A,-B,-C),(-A,B,-C)).execute(model, trans)

    def widget(self):
        return InfiniteVolumeGUI(self)

    def generateParameters(self, model, trans):
        a,b,c,A,B,C = trans(self.args, unit="m")
        ids = [-1, -1, -1, -1, -1, -1]
        for dim, grp in model.getPhysicalGroups(3):
            for tag in model.getEntitiesForPhysicalGroup(dim, grp):
                if model.isInside(dim, tag, [(a+A)/2, 0, 0]):
                    ids[0] = grp
                if model.isInside(dim, tag, [-(a+A)/2, 0, 0]):
                    ids[1] = grp
                if model.isInside(dim, tag, [0, (b+B)/2, 0]):
                    ids[2] = grp
                if model.isInside(dim, tag, [0, -(b+B)/2, 0]):
                    ids[3] = grp
                if model.isInside(dim, tag, [0, 0, (c+C)/2]):
                    ids[4] = grp
                if model.isInside(dim, tag, [0, 0, -(c+C)/2]):
                    ids[5] = grp
        J = {ids[i]: self._constructJ(i) for i in range(6)}
        J["default"] = np.eye(3)
        return {"J": FEMCoefficient(J)}

    def _constructJ(self, domain):
        a,b,c,A,B,C = np.array(self.args)
        alpha = sp.Integer(2)

        Cx = np.array([0, b-a*(B-b)/(A-a), c-a*(C-c)/(A-a)])
        Cy = np.array([a-b*(A-a)/(B-b), 0, c-b*(C-c)/(B-b)])
        Cz = np.array([a-c*(A-a)/(C-c), b-c*(B-b)/(C-c), 0])
        Cb = (Cx + Cy + Cz)/3

        x,y,z = sp.symbols("x,y,z")
        if domain == 0:
            X = Cb[0] + (a-Cb[0])/((A-x)/(A-a))**(1/alpha)
            Y = y * (X-Cy[0]) / (x-Cy[0])
            Z = z * (X-Cz[0]) / (x-Cz[0])
        elif domain == 1:
            X = -(Cb[0] + (a-Cb[0])/((A+x)/(A-a))**(1/alpha))
            Y = y * (-X-Cy[0]) / (-x-Cy[0])
            Z = z * (-X-Cz[0]) / (-x-Cz[0])
        elif domain == 2:
            Y = Cb[1] + (b-Cb[1])/((B-y)/(B-b))**(1/alpha)
            X = x * (Y-Cx[1]) / (y-Cx[1])
            Z = z * (Y-Cz[1]) / (y-Cz[1])
        elif domain == 3:
            Y = -(Cb[1] + (b-Cb[1])/((B+y)/(B-b))**(1/alpha))
            X = x * (-Y-Cx[1]) / (-y-Cx[1])
            Z = z * (-Y-Cz[1]) / (-y-Cz[1])
        elif domain == 4:
            Z = Cb[2] + (c-Cb[2])/((C-z)/(C-c))**(1/alpha)
            X = x * (Z -Cx[2]) / (z-Cx[2])
            Y = y * (Z -Cy[2]) / (z-Cy[2])
        elif domain == 5:
            Z = -(Cb[2] + (c-Cb[2])/((C+z)/(C-c))**(1/alpha))
            X = x * (-Z -Cx[2]) / (-z-Cx[2])
            Y = y * (-Z -Cy[2]) / (-z-Cy[2])
        m = np.array([[X.diff(x), Y.diff(x), Z.diff(x)],[X.diff(y), Y.diff(y), Z.diff(y)],[X.diff(z), Y.diff(z), Z.diff(z)]]) 
        det = m[0,0]*m[1,1]*m[2,2] + m[0,1]*m[1,2]*m[2,0] + m[0,2]*m[1,0]*m[2,1]
        J = sp.Matrix([
            [m[1,1]*m[2,2]-m[1,2]*m[2,1], m[0,2]*m[2,1]-m[0,1]*m[2,2], m[0,1]*m[1,2]-m[0,2]*m[1,1]],
            [m[1,2]*m[2,0]-m[1,0]*m[2,2], m[0,0]*m[2,2]-m[0,2]*m[2,0], m[0,2]*m[1,0]-m[0,0]*m[1,2]],
            [m[1,0]*m[2,1]-m[1,1]*m[2,0], m[0,1]*m[2,0]-m[0,0]*m[2,1], m[0,0]*m[1,1]-m[0,1]*m[1,0]]])/det
        return [[J[i,j] for j in range(3)] for i in range(3)]
    

class Rect(FEMGeometry):
    def __init__(self, x=0, y=0, z=0, dx=1, dy=1):
        super().__init__([x, y, z, dx, dy])

    def execute(self, model, trans):
        args = trans(self.args, unit="m")
        model.occ.addRectangle(*args)

    @classmethod
    @property
    def type(cls):
        return "rectangle"

    def widget(self):
        return RectGUI(self)


class Disk(FEMGeometry):
    def __init__(self, x=0, y=0, z=0, rx=1, ry=1):
        super().__init__([x, y, z, rx, ry])

    def execute(self, model, trans):
        args = trans(self.args, unit="m")
        model.occ.addDisk(*args)

    @classmethod
    @property
    def type(cls):
        return "disk"

    def widget(self):
        return DiskGUI(self)


class Quad(FEMGeometry):
    def __init__(self, v1=(0,0,0), v2=(1,0,0), v3=(1,1,0), v4=(0,1,0)):
        super().__init__([v1, v2 ,v3, v4])

    def execute(self, model, trans):
        pts = [model.occ.addPoint(*trans(v, unit="m")) for v in self.args]
        lines = [model.occ.addLine(pts[0], pts[1]), model.occ.addLine(pts[1], pts[2]), model.occ.addLine(pts[2], pts[3]), model.occ.addLine(pts[3], pts[0])]
        model.occ.addSurfaceFilling(model.occ.addCurveLoop([lines[0], lines[1], lines[2], lines[3]]))

    @classmethod
    @property
    def type(cls):
        return "quad"

    def widget(self):
        return QuadGUI(self)


class InfinitePlane(FEMGeometry):
    type = "infinite plane"
    def __init__(self, a=1, b=1, A=2, B=2):
        super().__init__([a,b,A,B])

    def execute(self, model, trans):
        a,b,A,B = self.args
        Quad((a,b,0), (-a,b,0), (-A,B,0),(A,B,0)).execute(model, trans)
        Quad((a,-b,0), (-a,-b,0), (-A,-B,0),(A,-B,0)).execute(model, trans)
        Quad((a,b,0), (a,-b,0), (A,-B,0),(A,B,0)).execute(model, trans)
        Quad((-a,b,0), (-a,-b,0), (-A,-B,0),(-A,B,0)).execute(model, trans)

    def generateParameters(self, model, trans):
        a,b,A,B = trans(self.args)
        ids = [-1, -1, -1, -1]
        for dim, grp in model.getPhysicalGroups(2):
            for tag in model.getEntitiesForPhysicalGroup(dim, grp):
                if model.isInside(dim, tag, [(a+A)/2, 0, 0]):
                    ids[0] = grp
                if model.isInside(dim, tag, [-(a+A)/2, 0, 0]):
                    ids[1] = grp
                if model.isInside(dim, tag, [0, (b+B)/2, 0]):
                    ids[2] = grp
                if model.isInside(dim, tag, [0, -(b+B)/2, 0]):
                    ids[3] = grp
        J = {ids[i]: self._constructJ(i) for i in range(4)}
        J["default"] = np.eye(2)
        return {"J": FEMCoefficient(J)}

    def _constructJ(self, domain):
        a,b,A,B = self.args
        alpha = sp.Integer(1)

        Cx = np.array([0, b-a*(B-b)/(A-a)])
        Cy = np.array([a-b*(A-a)/(B-b), 0])
        Cb = (Cx + Cy)/2

        x,y = sp.symbols("x,y")
        if domain == 0:
            X = Cb[0] + (a-Cb[0])/((A-x)/(A-a))**(1/alpha)
            Y = y * (X-Cy[0]) / (x-Cy[0])
        if domain == 1:
            X = -(Cb[0] + (a-Cb[0])/((A+x)/(A-a))**(1/alpha))
            Y = y * (-X-Cy[0]) / (-x-Cy[0])
        elif domain == 2:
            Y = Cb[1] + (b-Cb[1])/((B-y)/(B-b))**(1/alpha)
            X = x * (Y-Cx[1]) / (y-Cx[1])
        elif domain == 3:
            Y = -(Cb[1] + (b-Cb[1])/((B+y)/(B-b))**(1/alpha))
            X = x * (-Y-Cx[1]) / (-y-Cx[1])
        m = np.array([[X.diff(x), Y.diff(x)],[X.diff(y), Y.diff(y)]])
        det = m[0,0]*m[1,1]-m[0,1]*m[1,0]
        J = sp.Matrix([[m[1,1], -m[0,1]], [-m[1,0], m[0,0]]])/det
        return [[J[i,j] for j in range(2)] for i in range(2)]

    def widget(self):
        return InfinitePlaneGUI(self)
    

class Line(FEMGeometry):
    def __init__(self, x1=0, y1=0, z1=0, x2=1, y2=0, z2=0):
        super().__init__([x1, y1, z1, x2, y2, z2])

    def execute(self, model, trans):
        arg1 = trans(self.args[:3], unit="m")
        arg2 = trans(self.args[3:], unit="m")
        p1t = model.occ.addPoint(*arg1)
        p2t = model.occ.addPoint(*arg2)
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
