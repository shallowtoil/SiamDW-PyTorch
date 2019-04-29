from .siamfc import SiamFC_
from .siamrpn import SiamRPN_
from .features import ResNet22, Incep22
from .connect import Corr_Up
from lib.utils.config import configSiamFC, configSiamRPN

class SiamFC_Res22(SiamFC_):
    def __init__(self, tracking=True, **kwargs):
        super(SiamFC_Res22, self).__init__(**kwargs)
        self.tracking = tracking
        self.features = ResNet22(pretrain=configSiamFC.load_imagenet)
        self.connect_model = Corr_Up(tracking=self.tracking)

# class SiamFC_Incep22(SiamFC_):
#     def __init__(self, tracking=True, **kwargs):
#         super(SiamFC_Incep22, self).__init__(**kwargs)
#         self.tracking = tracking
#         self.features = Incep22()
#         self.connect_model = Corr_Up(self.tracking)

class SiamRPN_Res22(SiamRPN_):
    def __init__(self, tracking=True, **kwargs):
        super(SiamRPNRes22, self).__init__(**kwargs)
        self.tracking = tracking
        self.features = ResNet22(pretrain=configSiamRPN.load_imagenet)
        self.connect_model = RPN_Up(tracking=self.tracking,
                                    anchor_nums=self.anchor_nums,
                                    inchannels=self.features.feature_size,
                                    outchannels=256)







