import torch
import torch.nn as nn
import torch.nn.functional as F


class Corr_Up(nn.Module):
    def __init__(self, tracking=True):
        super(Corr_Up, self).__init__()
        self.tracking = tracking
        self.loc_adjust = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, z_f, x_f):
        # z_f exemplar small
        # x_f instance big
        if not self.tracking:
            # when train and valid
            N, _, H, W = x_f.shape
            x_f = x_f.view(1, -1, H, W)
            pred_loc = self.loc_adjust(F.conv2d(x_f, z_f, groups=N).transpose(0, 1))
        else:
            # when track
            pred_loc = self.loc_adjust(F.conv2d(x_f, z_f))

        return pred_loc


