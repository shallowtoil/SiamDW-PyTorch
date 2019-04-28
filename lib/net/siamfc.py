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
        self.tracking = None #default=True
        self.features = None #ResNet22()
        self.connect_model = None #Corr_Up()
        self.zf = None #set during training(z img of a pairs) and tracking(initial frame)

        if not self.tracking:
            # 这里要改一下的 裁成500以后 就不需要分255和231了 都做255处理
            gt, weight = self._create_gt_mask((config.train_score_size, config.train_score_size))
            # with torch.cuda.device(gpu_id):
            self.train_gt = torch.from_numpy(gt).cuda()
            self.train_weight = torch.from_numpy(weight).cuda()

            gt, weight = self._create_gt_mask((config.score_size, config.score_size))
            # with torch.cuda.device(gpu_id):
            self.valid_gt = torch.from_numpy(gt).cuda()
            self.valid_weight = torch.from_numpy(weight).cuda()

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
