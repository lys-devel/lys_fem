import numpy as np
import sympy as sp
from lys_fem import addGeometry, FEMGeometry
from lys_fem.fem import FEMCoefficient


class Box(FEMGeometry):
    type = "box"
    def __init__(self, x=0, y=0, z=0, dx=1, dy=1, dz=1):
        super().__init__([x, y, z, dx, dy, dz])

    def execute(self, model, trans):
        x,y,z,dx,dy,dz = self.args
        return RectFrustum((x,y,z), (x+dx,y,z), (x+dx,y+dy,z), (x,y+dy,z), (x,y,z+dz), (x+dx,y,z+dz), (x+dx, y+dy, z+dz), (x,y+dy, z+dz)).execute(model, trans)

    def widget(self):
        from .geometryGUI import BoxGUI
        return BoxGUI(self)


class Sphere(FEMGeometry):
    type = "sphere"
    def __init__(self, x=0, y=0, z=0, r=1):
        super().__init__([x, y, z, r])

    def execute(self, model, trans):
        x,y,z,r = trans(self.args, unit="m")

        p1 = model.geo.addPoint(x, y, z)
        p2 = model.geo.addPoint(x + r, y, z)
        p3 = model.geo.addPoint(x, y + r, z)
        p4 = model.geo.addPoint(x, y, z + r)
        p5 = model.geo.addPoint(x - r, y, z)
        p6 = model.geo.addPoint(x, y - r, z)
        p7 = model.geo.addPoint(x, y, z - r)

        c1 = model.geo.addCircleArc(p2, p1, p7)
        c2 = model.geo.addCircleArc(p7, p1, p5)
        c3 = model.geo.addCircleArc(p5, p1, p4)
        c4 = model.geo.addCircleArc(p4, p1, p2)
        c5 = model.geo.addCircleArc(p2, p1, p3)
        c6 = model.geo.addCircleArc(p3, p1, p5)
        c7 = model.geo.addCircleArc(p5, p1, p6)
        c8 = model.geo.addCircleArc(p6, p1, p2)
        c9 = model.geo.addCircleArc(p7, p1, p3)
        c10 =model.geo.addCircleArc(p3, p1, p4)
        c11 =model.geo.addCircleArc(p4, p1, p6)
        c12 =model.geo.addCircleArc(p6, p1, p7)
    
        s1 = model.geo.addSurfaceFilling([model.geo.addCurveLoop([c5, c10, c4])])
        s2 = model.geo.addSurfaceFilling([model.geo.addCurveLoop([c9, -c5, c1])])
        s3 = model.geo.addSurfaceFilling([model.geo.addCurveLoop([c12, -c8, -c1])])
        s4 = model.geo.addSurfaceFilling([model.geo.addCurveLoop([c8, -c4, c11])])
        s5 = model.geo.addSurfaceFilling([model.geo.addCurveLoop([-c10, c6, c3])])
        s6 = model.geo.addSurfaceFilling([model.geo.addCurveLoop([-c11, -c3, c7])])
        s7 = model.geo.addSurfaceFilling([model.geo.addCurveLoop([-c2, -c7, -c12])])
        s8 = model.geo.addSurfaceFilling([model.geo.addCurveLoop([-c6, -c9, c2])])

        sl = model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6, s7, s8])
        return model.geo.addVolume([sl])

    def widget(self):
        from .geometryGUI import SphereGUI
        return SphereGUI(self)
    

class RectFrustum(FEMGeometry):
    type = "rectngular frustum"
    def __init__(self, v1=(0,0,0), v2=(1,0,0), v3=(1,1,0), v4=(0,1,0), v5=(0,0,1), v6=(1,0,1), v7=(1,1,1), v8=(0,1,1)):
        super().__init__([v1, v2, v3, v4, v5, v6, v7, v8])

    def execute(self, model, trans):
        v = self.args
        s1 = Quad(v[0],v[1],v[2],v[3]).execute(model, trans)
        s2 = Quad(v[4],v[5],v[6],v[7]).execute(model, trans)
        s3 = Quad(v[0],v[1],v[5],v[4]).execute(model, trans)
        s4 = Quad(v[1],v[2],v[6],v[5]).execute(model, trans)
        s5 = Quad(v[2],v[3],v[7],v[6]).execute(model, trans)
        s6 = Quad(v[3],v[0],v[4],v[7]).execute(model, trans)
        
        return model.geo.addVolume([model.geo.addSurfaceLoop([s1, s3, s4, s5, s6, s2])])

    def widget(self):
        from .geometryGUI import RectFrustumGUI
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
        from .geometryGUI import InfiniteVolumeGUI
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
        alpha = sp.Integer(1)

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
        return [[str(J[i,j]) for j in range(3)] for i in range(3)]
    

class Rect(FEMGeometry):
    type = "rectangle"
    def __init__(self, x=0, y=0, z=0, dx=1, dy=1):
        super().__init__([x, y, z, dx, dy])

    def execute(self, model, trans):
        x,y,z,dx,dy = self.args
        return Quad((x,y,z), (x+dx,y,z), (x+dx,y+dy,z), (x,y+dy,z)).execute(model, trans)

    def widget(self):
        from .geometryGUI import RectGUI
        return RectGUI(self)


class Disk(FEMGeometry):
    type = "disk"
    def __init__(self, x=0, y=0, z=0, rx=1, ry=1):
        super().__init__([x, y, z, rx, ry])

    def execute(self, model, trans):
        x,y,z,rx, ry = trans(self.args, unit="m")

        p1 = model.geo.addPoint(x, y, z)
        p2 = model.geo.addPoint(x + rx, y, z)
        p3 = model.geo.addPoint(x, y + ry, z)
        p4 = model.geo.addPoint(x - rx, y, z)
        p5 = model.geo.addPoint(x, y - ry, z)

        c1 = model.geo.addEllipseArc(p2, p1, p2 if rx > ry else p3, p3)
        c2 = model.geo.addEllipseArc(p3, p1, p4 if rx > ry else p3, p4)
        c3 = model.geo.addEllipseArc(p4, p1, p4 if rx > ry else p5, p5)
        c4 = model.geo.addEllipseArc(p5, p1, p2 if rx > ry else p5, p2)
    
        return model.geo.addSurfaceFilling([model.geo.addCurveLoop([c1, c2, c3, c4])])

    def widget(self):
        from .geometryGUI import DiskGUI
        return DiskGUI(self)


class Quad(FEMGeometry):
    type = "quad"
    def __init__(self, v1=(0,0,0), v2=(1,0,0), v3=(1,1,0), v4=(0,1,0)):
        super().__init__([v1, v2 ,v3, v4])

    def execute(self, model, trans):
        pts = [model.geo.addPoint(*trans(v, unit="m")) for v in self.args]
        lines = [model.geo.addLine(pts[0], pts[1]), model.geo.addLine(pts[1], pts[2]), model.geo.addLine(pts[2], pts[3]), model.geo.addLine(pts[3], pts[0])]
        return model.geo.addPlaneSurface([model.geo.addCurveLoop([lines[0], lines[1], lines[2], lines[3]], reorient=True)])

    def widget(self):
        from .geometryGUI import QuadGUI
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
        return [[str(J[i,j]) for j in range(2)] for i in range(2)]

    def widget(self):
        from .geometryGUI import InfinitePlaneGUI
        return InfinitePlaneGUI(self)
    

class Line(FEMGeometry):
    type = "line"
    def __init__(self, x1=0, y1=0, z1=0, x2=1, y2=0, z2=0):
        super().__init__([x1, y1, z1, x2, y2, z2])

    def execute(self, model, trans):
        arg1 = trans(self.args[:3], unit="m")
        arg2 = trans(self.args[3:], unit="m")
        p1t = model.geo.addPoint(*arg1)
        p2t = model.geo.addPoint(*arg2)
        return model.geo.addLine(p1t, p2t)

    def widget(self):
        from .geometryGUI import LineGUI
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
