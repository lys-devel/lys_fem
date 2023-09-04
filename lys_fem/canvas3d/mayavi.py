from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from traitsui.api import View, Item
from traits.api import HasTraits, Instance, on_trait_change

from lys.Qt import QtGui, QtWidgets


class _Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=250, width=300, show_label=False), resizable=True)


class canvas3d(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__initlayout()
        self._mlab = self._visualization.scene.mlab

    def __initlayout(self):
        self._visualization = _Visualization()
        self._ui = self._visualization.edit_traits(parent=self, kind='subpanel').control

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._ui)
        self._ui.setParent(self)

    def setGeometry(self, geom):
        for m in geom.getMesh():
            surf = self._mlab.pipeline.surface(m)
        print(surf)
        #self._mlab.pipeline.surface(self._mlab.pipeline.extract_edges(surf), color=(0, 0, 0))
