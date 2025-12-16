from simtools.widgets import CommonFileParser, CommonFileSystemView, CommonFileDialog


class FEMFile(CommonFileParser):
    _name = "FEM"
    _file = "input.dic"


class FEMFileSystemView(CommonFileSystemView):
    def __init__(self, **args):
        super().__init__(FEMFile, **args)


class FEMFileDialog(CommonFileDialog):
    def __init__(self, parent):
        super().__init__(parent, FEMFileSystemView, FEMFile)
