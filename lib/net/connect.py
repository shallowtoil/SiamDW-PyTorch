import torch
import torch.nn as nn
import torch.nn.functional as F


class Corr_Up(nn.Module):
    """
    SiamFC head
    """
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

class RPN_Up(nn.Module):
    """
    SiamRPN head
    """
    def __init__(self, tracking=True, anchor_nums=5, in_channels=256, out_channels=256):
        super(RPN_Up, self).__init__()
        self.tracking = tracking
        self.anchor_nums = anchor_nums
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.cls_channel = self.anchor_nums
        self.reg_channel = 4 * self.anchor_nums

        self.template_cls = nn.Conv2d(self.in_channels, self.out_channels * self.cls_channel, kernel_size=3)
        self.template_reg = nn.Conv2d(self.in_channels, self.out_channels * self.reg_channel, kernel_size=3)

        self.search_cls = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3)
        self.search_reg = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3)
        self.loc_adjust = nn.Conv2d(self.reg_channel, self.reg_channel, kernel_size=1)

    def forward(self, z_f, x_f):
        # z_f exemplar small
        # x_f instance big
        cls_kernel = self.template_cls(z_f)
        reg_kernel = self.template_reg(z_f)

        cls_feature = search_cls(x_f)
        reg_feature = search_reg(x_f)

        if not self.tracking:
            # when train and valid
            N, _, H, W = x_f.shape
            x_f = x_f.view(1, -1, H, W)
            pred_cls = F.conv2d(cls_feature, cls_kernel, groups=N).transpose(0,1)
            pred_reg = self.loc_adjust(F.conv2d(reg_feature, reg_kernel, groups=N).transpose(0,1))
        else:
            # when track
            pred_cls = F.conv2d(cls_feature, cls_kernel)
            pred_reg = self.loc_adjust(F.conv2d(reg_feature, reg_kernel))

        return pred_cls, pred_reg