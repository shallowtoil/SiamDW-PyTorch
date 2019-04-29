import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

import sys
import os
sys.path.append(os.getcwd())
from lib.utils.config import config

class SiamFC_(nn.Module):
    def __init__(self):
        super(SiamFC_, self).__init__()
        self.features = None #ResNet22()
        self.connect_model = None #Corr_Up()
        self.zf = None #set during training(z img of a pairs) and tracking(initial frame)

    def feature_extractor(self, x):
        return self.features(x)

    def connector(self, template_feature, search_feature):
        pred_score = self.connect_model(template_feature, search_feature)
        return pred_score

    def template(self, z):
        self.zf = self.feature_extractor(z)
        # [batch_size,512,5,5]

    def track(self, x):
        xf = self.feature_extractor(x)
        # [batch_size,512,19,19]
        pred_score = self.connector(self.zf, xf)
        # [batch_size,1,15,15]
        return pred_score
