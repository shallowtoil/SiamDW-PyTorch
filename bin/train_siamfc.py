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
from lib.utils.config import configSiamFC as config
import lib.net.models as models
from lib.dataset.dataset import SiamFCDataset
from lib.dataset.custom_transforms import ToTensor, RandomStretch, \
    RandomCrop, CenterCrop
from lib.utils.loss import weighted_binary_cross_entropy

torch.manual_seed(1234)

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--arch', dest='arch', default='SiamFC_Res22', help='architecture of to-be-trained model')
parser.add_argument('--resume', default='CIResNet22', type=str, help='pretrained model')

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
    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        RandomCrop((config.instance_size, config.instance_size),
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
    train_dataset = SiamFCDataset(db, train_videos, data_dir,
                                      train_z_transforms, train_x_transforms, training=True)
    valid_dataset = SiamFCDataset(db, valid_videos, data_dir,
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

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    for epoch in range(1, config.epoch + 1):
        if config.fix_former_layers:
            model.freeze_layers()
        train_loss = []
        model.train()
        tic = time.clock()

        loss_temp = 0
        for i, data in enumerate(trainloader):
            exemplar_imgs, instance_imgs, masks, weights = data
            exemplar_var, instance_var = Variable(exemplar_imgs.cuda()), \
                    Variable(instance_imgs.cuda())
            masks, weights = masks.cuda(), weights.cuda()
            optimizer.zero_grad()
            model.template(exemplar_var) #[bz,3,127,127]->[bz,1,5,5]
            outputs = model.track(instance_var) #[bz,3,239,239]->[bz,1,19,19]
            loss = weighted_binary_cross_entropy(outputs, masks, weights)
            loss.backward()
            optimizer.step()

            step = (epoch - 1) * len(trainloader) + i
            summary_writer.add_scalar('train/loss', loss.data, step)
            train_loss.append(loss.detach().cpu())
            loss_temp += cls_loss.detach().cpu().numpy()
            if (i + 1) % config.show_interval == 0:
                print("[epoch %2d][iter %4d], loss: %.4f" %
                      (epoch, i + 1, loss_temp / config.show_interval))
                loss_temp = 0
        train_loss = np.mean(train_loss)
        toc = time.clock() - tic
        print('%ss total for one epoch' % toc)


        valid_loss = []
        model.eval()
        for i, data in enumerate(validloader):
            exemplar_imgs, instance_imgs = data
            exemplar_var, instance_var = Variable(exemplar_imgs.cuda()),\
                                         Variable(instance_imgs.cuda())
            model.template(exemplar_var) #[bz,3,127,127]->[bz,1,5,5]
            outputs = model.track(instance_var) #[bz,3,255,255]->[bz,1,21,21]
            loss = model.weighted_loss(outputs) #[bz,1,17,17]
            valid_loss.append(loss.data.item())
        valid_loss = np.mean(valid_loss)
        print("[epoch %d] valid_loss: %.4f, train_loss: %.4f" % (epoch, valid_loss, train_loss))
        summary_writer.add_scalar('valid/loss', 
                valid_loss, (epoch + 1) * len(trainloader))
        if epoch % config.save_interval == 0:
            if not os.path.exists('./models/'):
                os.makedirs("./models/")
            save_name = "./models/" + args.resume + "_{}.pth".format(epoch)
            new_state_dict = model.state_dict()
            if torch.cuda.device_count() > 1:
                new_state_dict = OrderedDict()
                for k, v in model.state_dict().items():
                    namekey = k[7:]  # remove `module.`
                    new_state_dict[namekey] = v
            torch.save(new_state_dict, save_name)
            print('save model: {}'.format(save_name))
        model.cuda()
        scheduler.step()


if __name__ == '__main__':
    train()
