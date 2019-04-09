from .siamfc import SiamFC_
from .features import ResNet22, Incep22
from .connect import Corr_Up


class SiamFC_Res22(SiamFC_):
    def __init__(self, imagenet=False, freeze=False, tracking=True, **kwargs):
        super(SiamFC_Res22, self).__init__(**kwargs)
        self.tracking = tracking
        self.imagenet = imagenet
        self.freeze = freeze
        self.features = ResNet22(self.imagenet, self.freeze)
        self.connect_model = Corr_Up(self.tracking)

class SiamFC_Incep22(SiamFC_):
    def __init__(self, tracking=True, **kwargs):
        super(SiamFC_Incep22, self).__init__(**kwargs)
        self.tracking = tracking
        self.features = Incep22()
        self.connect_model = Corr_Up(self.tracking)








