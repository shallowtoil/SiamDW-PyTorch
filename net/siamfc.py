import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from utils.config import config

class SiamFC_(nn.Module):
    def __init__(self):
        super(SiamFC_, self).__init__()
        self.tracking = None #default=True
        self.features = None #ResNet22()
        self.connect_model = None #Corr_Up()
        self.zf = None #set during training(z img of a pairs) and tracking(initial frame)

        if not self.tracking:
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

    def weighted_loss(self, pred):
        # with backprop
        if not self.tracking and self.training:
            return F.binary_cross_entropy_with_logits(pred, self.train_gt,
                                                      self.train_weight)
                                                      # reduction='sum') / config.train_batch_size # normalize the batch_size
        # without backprop
        elif not self.tracking and not self.training:
            return F.binary_cross_entropy_with_logits(pred, self.valid_gt,
                                                      self.valid_weight)
                                                      #reduction='sum') / config.valid_batch_size # normalize the batch_size

    def _create_gt_mask(self, shape):
        # same for all pairs
        h, w = shape
        y = np.arange(h, dtype=np.float32) - (h-1) / 2.
        x = np.arange(w, dtype=np.float32) - (w-1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x**2 + y**2)
        mask = np.zeros((h, w))
        mask[dist <= config.radius / config.total_stride] = 1
        mask = mask[np.newaxis, :, :]
        weights = np.ones_like(mask)
        weights[mask == 1] = 0.5 / np.sum(mask == 1)
        weights[mask == 0] = 0.5 / np.sum(mask == 0)
        mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
        return mask.astype(np.float32), weights.astype(np.float32)