
from lys_fem import addGeometry, FEMGeometry
from .geometryGUI import BoxGUI, RectGUI, LineGUI, DiskGUI


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


class Line(FEMGeometry):
    def __init__(self, x1=0, y1=0, z1=0, x2=1, y2=0, z2=0):
        self.args = (x1, y1, z1, x2, y2, z2)

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
addGeometry("Add 2D", Rect)
addGeometry("Add 2D", Disk)
addGeometry("Add 1D", Line)
