import weakref
import gmsh
import numpy as np
import sympy as sp
from lys_fem import util

geometryCommands = {}


def addGeometry(group, geom):
    if group not in geometryCommands:
        geometryCommands[group] = []
    geometryCommands[group].append(geom)


class FEMGeometry(object):
    type = "invalid"
    """:meta private:"""

    def __init__(self, args):
        self._args = args
        self._method = None

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value
        if self._method is not None:
            if self._method() is not None:
                self._method()()

    def setCallback(self, method):
        self._method = weakref.ref(method)

    def saveAsDictionary(self):
        return {"type": self.type, "args": self.args}

    @staticmethod
    def loadFromDictionary(d):
        for t in sum(geometryCommands.values(), []):
            if t.type == d["type"]:
                return t(*d["args"])

    def generateParameters(self, model, scale):
        return {}

    def execute(self, model, trans):
        """
        :meta private:
        Execute occ mesher command. This method should be overwritten by the inherited class.
        """
        raise NotImplementedError("[execute] method should be implemented.")

    def widget(self):
        """
        :meta private:
        Return a widget for this geometry component.
        """
        raise NotImplementedError("[widget] method should be implemented.")


class Box(FEMGeometry):
    """
    Three-dimensional box geometry.

    Args:
        x,y,z (float): The origin of the box.
        dx, dy, dz (float): The size of the box.
    """
    type = "box"
    def __init__(self, x=0, y=0, z=0, dx=1, dy=1, dz=1):
        super().__init__([x, y, z, dx, dy, dz])

    def execute(self, model, trans):
        args = trans(self.args, unit="m")
        model.occ.addBox(*args)

    def widget(self):
        from .geometryGUI import BoxGUI
        return BoxGUI(self)


class Sphere(FEMGeometry):
    """
    Three-dimensional sphere geometry.

    Args:
        x,y,z (float): The origin of the sphere.
        r (float): The radius of the sphere.
    """
    type = "sphere"
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

    def widget(self):
        from .geometryGUI import SphereGUI
        return SphereGUI(self)
    

class RectFrustum(FEMGeometry):
    type = "rectngular frustum"
    def __init__(self, v1=(0,0,0), v2=(1,0,0), v3=(1,1,0), v4=(0,1,0), v5=(0,0,1), v6=(1,0,1), v7=(1,1,1), v8=(0,1,1)):
        super().__init__([v1, v2, v3, v4, v5, v6, v7, v8])

    def execute(self, model, trans):
        pts = [model.occ.addPoint(*trans(v, unit="m")) for v in self.args]

        lines = [model.occ.addLine(pts[0], pts[1]), model.occ.addLine(pts[1], pts[2]), model.occ.addLine(pts[2], pts[3]), model.occ.addLine(pts[3], pts[0])]
        lines.extend([model.occ.addLine(pts[4], pts[5]), model.occ.addLine(pts[5], pts[6]), model.occ.addLine(pts[6], pts[7]), model.occ.addLine(pts[7], pts[4])])      
        lines.extend([model.occ.addLine(pts[0], pts[4]), model.occ.addLine(pts[1], pts[5]), model.occ.addLine(pts[2], pts[6]), model.occ.addLine(pts[3], pts[7])])  

        s1 = model.occ.addPlaneSurface([model.occ.addCurveLoop([lines[0], lines[1], lines[2], lines[3]])])
        s2 = model.occ.addPlaneSurface([model.occ.addCurveLoop([lines[4], lines[5], lines[6], lines[7]])])
        s3 = model.occ.addPlaneSurface([model.occ.addCurveLoop([lines[0], lines[8], lines[4], lines[9]])])
        s4 = model.occ.addPlaneSurface([model.occ.addCurveLoop([lines[1], lines[9], lines[5], lines[10]])])
        s5 = model.occ.addPlaneSurface([model.occ.addCurveLoop([lines[2], lines[10], lines[6], lines[11]])])
        s6 = model.occ.addPlaneSurface([model.occ.addCurveLoop([lines[3], lines[11], lines[7], lines[8]])])
        
        model.occ.addVolume([model.occ.addSurfaceLoop([s1, s3, s4, s5, s6, s2])])

    def isInside(self, r):
        v = self.args
        c = np.mean(v, axis=0)
        np1 = self._normalVector(v[0], v[1], v[2], c)
        np2 = self._normalVector(v[4], v[5], v[6], c)
        np3 = self._normalVector(v[0], v[1], v[4], c)
        np4 = self._normalVector(v[1], v[2], v[5], c)
        np5 = self._normalVector(v[2], v[3], v[6], c)
        np6 = self._normalVector(v[0], v[3], v[4], c)
        return all(np.dot(n, r - p) < 0 for n, p in [np1, np2, np3, np4, np5, np6])
    
    def _normalVector(self, a, b, c, center):
        a,b,c,center = np.array([a,b,c,center])
        n = np.cross(b-a, c-a)
        n = n/np.linalg.norm(n)
        p = np.mean([a,b,c], axis=0)
        if np.dot(n, center - p) > 0:
            n = -n
        return n, p

    def widget(self):
        from .geometryGUI import RectFrustumGUI
        return RectFrustumGUI(self)
    

class InfiniteVolume(FEMGeometry):
    type = "infinite volume"
    def __init__(self, a=1, b=1, c=1, A=2, B=2, C=2):
        super().__init__([a,b,c,A,B,C])
        self._rf_zp = RectFrustum((a,b,c), (-a,b,c), (-a,-b,c),(a,-b,c),(A,B,C), (-A,B,C), (-A,-B,C),(A,-B,C))
        self._rf_zn = RectFrustum((a,b,-c), (-a,b,-c), (-a,-b,-c),(a,-b,-c),(A,B,-C), (-A,B,-C), (-A,-B,-C),(A,-B,-C))
        self._rf_yp = RectFrustum((a,b,c), (-a,b,c), (-a,b,-c),(a,b,-c),(A,B,C), (-A,B,C), (-A,B,-C),(A,B,-C))
        self._rf_yn = RectFrustum((a,-b,c), (-a,-b,c), (-a,-b,-c),(a,-b,-c),(A,-B,C), (-A,-B,C), (-A,-B,-C),(A,-B,-C))
        self._rf_xp = RectFrustum((a,b,c), (a,-b,c), (a,-b,-c),(a,b,-c),(A,B,C), (A,-B,C), (A,-B,-C),(A,B,-C))
        self._rf_xn = RectFrustum((-a,b,c), (-a,-b,c), (-a,-b,-c),(-a,b,-c),(-A,B,C), (-A,-B,C), (-A,-B,-C),(-A,B,-C))

    def execute(self, model, trans):
        self._rf_zp.execute(model, trans)
        self._rf_zn.execute(model, trans)
        self._rf_yp.execute(model, trans)
        self._rf_yn.execute(model, trans)
        self._rf_xp.execute(model, trans)
        self._rf_xn.execute(model, trans)

    def widget(self):
        from .geometryGUI import InfiniteVolumeGUI
        return InfiniteVolumeGUI(self)

    def generateParameters(self, model, trans):
        J = {"default": np.eye(3)}
        for dim, grp in model.getPhysicalGroups(3):
            for tag in model.getEntitiesForPhysicalGroup(dim, grp):
                p = trans.inv(model.occ.getCenterOfMass(dim, tag), unit="m")
                if self._rf_xp.isInside(p):
                    J[grp] = self._constructJ(0)
                if self._rf_xn.isInside(p):
                    J[grp] = self._constructJ(1)
                if self._rf_yp.isInside(p):
                    J[grp] = self._constructJ(2)
                if self._rf_yn.isInside(p):
                    J[grp] = self._constructJ(3)
                if self._rf_zp.isInside(p):
                    J[grp] = self._constructJ(4)
                if self._rf_zn.isInside(p):
                    J[grp] = self._constructJ(5)
        return {"J": util.eval(J, name="J", geom="domain")}
    
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
        return [[str(J[i,j]) for j in range(3)] for i in range(3)]
    

class Rect(FEMGeometry):
    type = "rectangle"
    def __init__(self, x=0, y=0, z=0, dx=1, dy=1):
        super().__init__([x, y, z, dx, dy])

    def execute(self, model, trans):
        args = trans(self.args, unit="m")
        model.occ.addRectangle(*args)

    def widget(self):
        from .geometryGUI import RectGUI
        return RectGUI(self)


class Disk(FEMGeometry):
    type = "disk"
    def __init__(self, x=0, y=0, z=0, rx=1, ry=1):
        super().__init__([x, y, z, rx, ry])

    def execute(self, model, trans):
        args = trans(self.args, unit="m")
        model.occ.addDisk(*args)

    def widget(self):
        from .geometryGUI import DiskGUI
        return DiskGUI(self)


class Quad(FEMGeometry):
    type = "quad"
    def __init__(self, v1=(0,0,0), v2=(1,0,0), v3=(1,1,0), v4=(0,1,0)):
        super().__init__([v1, v2 ,v3, v4])

    def execute(self, model, trans):
        pts = [model.occ.addPoint(*trans(v, unit="m")) for v in self.args]
        lines = [model.occ.addLine(pts[0], pts[1]), model.occ.addLine(pts[1], pts[2]), model.occ.addLine(pts[2], pts[3]), model.occ.addLine(pts[3], pts[0])]
        model.occ.addPlaneSurface([model.occ.addCurveLoop([lines[0], lines[1], lines[2], lines[3]])])

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
        a,b,A,B = trans(self.args, unit="m")

        J = {"default": np.eye(2)}
        for dim, grp in model.getPhysicalGroups(2):
            for tag in model.getEntitiesForPhysicalGroup(dim, grp):
                p = model.occ.getCenterOfMass(dim, tag)
                if self._checkInside(p, (a,b,0), (a,-b,0), (A,-B,0),(A,B,0)):
                    J[grp] = self._constructJ(0)
                if self._checkInside(p, (-a,b,0), (-a,-b,0), (-A,-B,0),(-A,B,0)):
                    J[grp] = self._constructJ(1)
                if self._checkInside(p, (a,b,0), (-a,b,0), (-A,B,0),(A,B,0)):
                    J[grp] = self._constructJ(2)
                if self._checkInside(p, (a,-b,0), (-a,-b,0), (-A,-B,0),(A,-B,0)):
                    J[grp] = self._constructJ(3)
        return {"J": util.eval(J, name="J", geom="domain")}
    
    def _checkInside(self, p, a, b, c, d):
        verts = np.array([a, b, c, d])
        c = np.mean(verts, axis=0)
        verts = sorted(verts, key=lambda p: np.arctan2(p[1] - c[1], p[0] - c[0]))

        s = []
        for i in range(4):
            v0 = verts[i]
            v1 = verts[(i+1) % 4]
            s.append(np.cross(v1 - v0, p - v0)[2])

        # allow boundary: treat near-zero as zero
        pos = any(x > 0 for x in s)
        neg = any(x < 0 for x in s)
        return not (pos and neg)  # not both signs => inside or on edge

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
        p1t = model.occ.addPoint(*arg1)
        p2t = model.occ.addPoint(*arg2)
        model.occ.addLine(p1t, p2t)

    def widget(self):
        from .geometryGUI import LineGUI
        return LineGUI(self)


class ImportGmsh(FEMGeometry):
    type = "gmsh(.msh)"
    def __init__(self, file):
        super().__init__([])
        self._file = file

    def execute(self, model, trans):
        gmsh.merge(self._file)

    def widget(self):
        from .geometryGUI import BoxGUI
        return BoxGUI(self)

addGeometry("Add 3D", Box)
addGeometry("Add 3D", Sphere)
addGeometry("Add 3D", RectFrustum)
addGeometry("Add 3D", InfiniteVolume)
addGeometry("Add 2D", Rect)
addGeometry("Add 2D", Disk)
addGeometry("Add 2D", Quad)
addGeometry("Add 2D", InfinitePlane)
addGeometry("Add 1D", Line)
addGeometry("Import", ImportGmsh)