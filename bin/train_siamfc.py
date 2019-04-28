#trainv2 = train with random shift
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
from lib.utils.config import config
import lib.net.models as models
from lib.dataset.dataset import ImagnetVIDDataset
from lib.dataset.custom_transforms import ToTensor, RandomStretch, \
    RandomCrop, CenterCrop

torch.manual_seed(1234)

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--arch', dest='arch', default='SiamFC_Res22', help='architecture of to-be-trained model')
parser.add_argument('--resume', default='./models/CIResNet22.pth', type=str, help='pretrained model')

def train():
    # initialize config
    global args
    args = parser.parse_args()

    # loading meta data
    data_dir = config.data_dir
    meta_data_path = os.path.join(data_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path,'rb'))
    all_videos = [x[0] for x in meta_data]

    # split train/valid dataset
    train_videos, valid_videos = train_test_split(all_videos, 
            test_size=1-config.train_ratio, random_state=config.seed)

    # define transforms
    random_crop_size = config.instance_size - 2 * config.total_stride
    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        RandomCrop((random_crop_size, random_crop_size),
                    config.max_translate),
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # open lmdb
    db = lmdb.open(data_dir+'.lmdb', readonly=True, map_size=int(50e9))

    # create dataset
    train_dataset = ImagnetVIDDataset(db, train_videos, data_dir,
            train_z_transforms, train_x_transforms)
    valid_dataset = ImagnetVIDDataset(db, valid_videos, data_dir,
            valid_z_transforms, valid_x_transforms, training=False)
    
    # create dataloader
    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
            shuffle=True, pin_memory=True, num_workers=config.train_num_workers, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
            shuffle=False, pin_memory=True, num_workers=config.valid_num_workers, drop_last=True)

    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)


    # start training + validation
    model = models.__dict__[args.arch](imagenet=True, freeze=False, tracking=False)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
            momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=config.gamma)
    # scheduler = StepLR(optimizer, step_size=config.step_size,
    #         gamma=config.gamma)

    for epoch in range(config.epoch):
        train_loss = []
        model.train()
        tic = time.clock()
        for i, data in enumerate(trainloader):
            exemplar_imgs, instance_imgs = data
            exemplar_var, instance_var = Variable(exemplar_imgs.cuda()), \
                    Variable(instance_imgs.cuda())
            optimizer.zero_grad()
            model.template(exemplar_var) #[bz,3,127,127]->[bz,1,5,5]
            outputs = model.track(instance_var) #[bz,3,239,239]->[bz,1,19,19]
            loss = model.weighted_loss(outputs) #[bz,1,15,15]
            loss.backward()
            optimizer.step()
            step = epoch * len(trainloader) + i
            summary_writer.add_scalar('train/loss', loss.data, step)
            if (i+1) % 20 == 0:
                print("EPOCH %d STEP %d, loss: %.4f" %
                      (epoch, i+1, loss.data))
            train_loss.append(loss.data.item())
        train_loss = np.mean(train_loss)
        toc = time.clock() - tic
        print('%ss total for one epoch' % toc)


        valid_loss = []
        model.eval()
        for i, data in enumerate(tqdm(validloader)):
            exemplar_imgs, instance_imgs = data
            exemplar_var, instance_var = Variable(exemplar_imgs.cuda()),\
                                         Variable(instance_imgs.cuda())
            model.template(exemplar_var) #[bz,3,127,127]->[bz,1,5,5]
            outputs = model.track(instance_var) #[bz,3,255,255]->[bz,1,21,21]
            loss = model.weighted_loss(outputs) #[bz,1,17,17]
            valid_loss.append(loss.data.item())
        valid_loss = np.mean(valid_loss)
        print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" %
                (epoch, valid_loss, train_loss))
        summary_writer.add_scalar('valid/loss', 
                valid_loss, (epoch+1)*len(trainloader))
        torch.save(model.cpu().state_dict(), args.resume)
        model.cuda()
        scheduler.step()


if __name__ == '__main__':
    train()
