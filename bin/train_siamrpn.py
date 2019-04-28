# trainv2 = train with random shift
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import time
import pickle
import lmdb
import argparse
import sys

from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

sys.path.append(os.getcwd())
from lib.utils.config import configSiamRPN as config
import lib.net.models as models
from lib.dataset.dataset import SiamRPNDataset
from lib.dataset.custom_transforms import ToTensor_with_bbox, RandomStretch_with_bbox, \
    RandomCrop_with_bbox, CenterCrop

torch.manual_seed(1234)

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--arch', dest='arch', default='SiamRPN_Res22', help='architecture of to-be-trained model')
parser.add_argument('--resume', default='./models/CIResNet22.pth', type=str, help='pretrained model')


def train():
    # initialize config
    global args
    args = parser.parse_args()

    # loading meta data
    data_dir = config.data_dir
    meta_data_path = os.path.join(data_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    all_videos = [x[0] for x in meta_data]

    # split train/valid dataset
    train_videos, valid_videos = train_test_split(all_videos,
                                                  test_size=1 - config.train_ratio, random_state=config.seed)

    # define transforms
    train_z_transforms = transforms.Compose([
        RandomStretch_with_bbox(),
        CenterCrop_with_bbox((config.exemplar_size, config.exemplar_size)),
        ToTensor_with_bbox()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch_with_bbox(),
        RandomCrop_with_bbox((config.instance_size, config.instance_size), config.max_translate),
        ToTensor_with_bbox()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # open lmdb
    db = lmdb.open(data_dir + '.lmdb', readonly=True, map_size=int(50e9))

    # create dataset
    train_dataset = SiamRPNDataset(db, train_videos, data_dir,
                                      train_z_transforms, train_x_transforms, training=true)
    anchors = train_dataset.anchors
    valid_dataset = SiamRPNDataset(db, valid_videos, data_dir,
                                      valid_z_transforms, valid_x_transforms, training=False)

    # create dataloader
    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=config.train_num_workers, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
                             shuffle=False, pin_memory=True,
                             num_workers=config.valid_num_workers, drop_last=True)

    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)

    # start training + validation
    model = models.__dict__[args.arch](tracking=False)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=config.gamma)
    # scheduler = StepLR(optimizer, step_size=config.step_size,
    #         gamma=config.gamma)

    for epoch in range(1, config.epoch + 1):
        train_loss = []
        model.train()
        tic = time.clock()

        loss_temp_cls = 0
        loss_temp_reg = 0
        for i, data in enumerate(trainloader):
            exemplar_imgs, instance_imgs, regression_target, conf_target = data
            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()
            exemplar_var, instance_var = Variable(exemplar_imgs.cuda()), \
                                         Variable(instance_imgs.cuda())
            optimizer.zero_grad()
            model.template(exemplar_var)  # [bz,3,127,127]->[bz,1,5,5]
            pred_cls, pred_reg = model.track(instance_var)  # [bz,3,239,239]->[bz,1,19,19]
            pred_conf = pred_cls.reshape(-1, 2, config.anchor_num * config.score_size * \
                                           config.score_size).permute(0, 2, 1)
            pred_offset = pred_reg.reshape(-1, 4, config.anchor_num * config.score_size * \
                                                  config.score_size).permute(0, 2, 1)
            cls_loss = rpn_cross_entropy_balance(pred_conf, conf_target, config.num_pos, config.num_neg, anchors,
                                                 ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
            reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target, config.num_pos, ohem=config.ohem_reg)
            loss = cls_loss + config.lamb * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()

            step = (epoch - 1) * len(trainloader) + i
            summary_writer.add_scalar('train/cls_loss', cls_loss.data, step)
            summary_writer.add_scalar('train/reg_loss', reg_loss.data, step)
            train_loss.append(loss.detach().cpu())
            loss_temp_cls += cls_loss.detach().cpu().numpy()
            loss_temp_reg += reg_loss.detach().cpu().numpy()
            if (i + 1) % config.show_interval == 0:
                print("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f" %
                      (epoch, i + 1, cls_loss.data / config.show_interval, reg_loss / config.show_interval))
                loss_temp_cls = 0
                loss_temp_reg = 0
            train_loss.append(loss.data.item())
        train_loss = np.mean(train_loss)
        toc = time.clock() - tic
        print('%ss total for one epoch' % toc)

        valid_loss = []
        model.eval()
        for i, data in enumerate(validloader):
            exemplar_imgs, instance_imgs, regression_target, conf_target = data
            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()
            exemplar_var, instance_var = Variable(exemplar_imgs.cuda()), \
                                         Variable(instance_imgs.cuda())
            model.template(exemplar_var)  # [bz,3,127,127]->[bz,1,5,5]
            pred_cls, pred_reg = model.track(instance_var)  # [bz,3,239,239]->[bz,1,19,19]
            pred_conf = pred_cls.reshape(-1, 2, config.anchor_num * config.score_size * \
                                         config.score_size).permute(0, 2, 1)
            pred_offset = pred_reg.reshape(-1, 4, config.anchor_num * config.score_size * \
                                           config.score_size).permute(0, 2, 1)
            cls_loss = rpn_cross_entropy_balance(pred_conf, conf_target, config.num_pos, config.num_neg, anchors,
                                                 ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
            reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target, config.num_pos, ohem=config.ohem_reg)
            loss = cls_loss + config.lamb * reg_loss
            valid_loss.append(loss.detach().cpu())
        valid_loss = np.mean(valid_loss)
        print("[epoch %d] valid_loss: %.4f, train_loss: %.4f" %
              (epoch, valid_loss, train_loss))
        summary_writer.add_scalar('valid/loss',
                                  valid_loss, (epoch + 1) * len(trainloader))

        torch.save(model.cpu().state_dict(), args.resume)
        model.cuda()
        scheduler.step()

if __name__ == '__main__':
    train()
