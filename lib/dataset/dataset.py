import cv2
import os
import numpy as np
import pickle
import hashlib
from torch.utils.data.dataset import Dataset
from lib.utils.config import configSiamFC, configSiamRPN
from lib.dataset.generate_target import generate_target, generate_scoremap
from lib.utils.utils import box_transform, compute_iou

class SiamFCDataset(Dataset):
    def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
        config = configSiamFC
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]:x[1] for x in self.meta_data}
        self.training = training
        # filter traj len less than 2
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

        self.txn = db.begin(write=False)
        self.num = len(self.video_names) if config.num_per_epoch is None or not self.training\
                else config.num_per_epoch
        self.scoremap = generate_scoremap((config.score_size, config.score_size))

    def imread(self, path):
        key = hashlib.md5(path.encode()).digest()
        img_buffer = self.txn.get(key)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def __getitem__(self, idx):
        idx = idx % len(self.video_names)
        video = self.video_names[idx]
        trajs = self.meta_data[video]
        # sample one trajs
        trkid = np.random.choice(list(trajs.keys()))
        traj = trajs[trkid]
        assert len(traj) > 1, "video_name: {}".format(video)
        # sample exemplar
        exemplar_idx = np.random.choice(list(range(len(traj))))
        exemplar_name = os.path.join(self.data_dir, video, traj[exemplar_idx]+".{:02d}.x.jpg".format(trkid))
        exemplar_img = self.imread(exemplar_name)
        exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
        # sample instance
        low_idx = max(0, exemplar_idx - config.frame_range)
        up_idx = min(len(traj), exemplar_idx + config.frame_range)
        # create sample weight, if the sample are far away from center
        # the probability being choosen are high
        weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)
        instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx+1:up_idx], p=weights)
        instance_name = os.path.join(self.data_dir, video, instance+".{:02d}.x.jpg".format(trkid))
        instance_img = self.imread(instance_name)
        instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
        if np.random.rand(1) < config.gray_ratio:
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
        # data augmentation
        exemplar_img = self.z_transforms(exemplar_img)
        instance_img = self.x_transforms(instance_img)
        # create target
        mask, weight = self.scoremap

        return exemplar_img, instance_img, mask, weight

    def __len__(self):
        return self.num


class SiamRPNDataset(Dataset):
    def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
        config = configSiamRPN
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]:x[1] for x in self.meta_data}
        self.training = training
        self.tracker = tracker
        # filter traj len less than 2
        for key in self.meta_data.keys():
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

        self.txn = db.begin(write=False)

        self.num = len(self.video_names) if config.pairs_per_video_per_epoch is None or not training \
            else config.pairs_per_video_per_epoch * len(self.video_names)
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        valid_scope)

    def imread(self, path):
        key = hashlib.md5(path.encode()).digest()
        img_buffer = self.txn.get(key)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def compute_target(self, anchors, box):
        regression_target = box_transform(anchors, box)
        iou = compute_iou(anchors, box).flatten()
        pos_index = np.where(iou > config.pos_threshold)[0]
        neg_index = np.where(iou < config.neg_threshold)[0]
        label = np.ones_like(iou) * -1
        label[pos_index] = 1
        label[neg_index] = 0
        return regression_target, label

    def __getitem__(self, idx):
        while True:
            idx = idx % len(self.video_names)
            video = self.video_names[idx]
            trajs = self.meta_data[video]
            # sample one trajs
            if len(trajs.keys()) > 0:
                break
            else:
                idx = np.random.randint(self.num)
        # sample one trajs
        trkid = np.random.choice(list(trajs.keys()))
        traj = trajs[trkid]
        assert len(traj) > 1, "video_name: {}".format(video)
        # sample exemplar
        exemplar_idx = np.random.choice(list(range(len(traj))))
        if 'ILSVRC2015' in video:
            exemplar_name = \
                glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))[0]
        else:
            exemplar_name = \
                glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{}.x*.jpg".format(trkid)))[0]
        exemplar_img = self.imread(exemplar_name)
        exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB) #TODO: have or have not?
        # sample instance
        if 'ILSVRC2015' in exemplar_name:
            frame_range = config.frame_range_vid
        else:
            frame_range = config.frame_range_ytb
        low_idx = max(0, exemplar_idx - frame_range)
        up_idx = min(len(traj), exemplar_idx + frame_range + 1)
        # create sample weight, if the sample are far away from center
        # the probability being choosen are high
        weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)
        instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx + 1:up_idx], p=weights)
        if 'ILSVRC2015' in video:
            instance_name = glob.glob(os.path.join(self.data_dir, video, instance + ".{:02d}.x*.jpg".format(trkid)))[0]
        else:
            instance_name = glob.glob(os.path.join(self.data_dir, video, instance + ".{}.x*.jpg".format(trkid)))[0]
        instance_img = self.imread(instance_name)
        instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB) #TODO: have or have not?
        gt_w, gt_h = float(instance_name.split('_')[-2]), float(instance_name.split('_')[-1][:-4])
        if np.random.rand(1) < config.gray_ratio:
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
        # data augmentation
        if config.exem_stretch:
            exemplar_img, _, _ = self.RandomStretch(exemplar_img, 0, 0)
        exemplar_img, _ = crop_and_pad(exemplar_img, (exemplar_img.shape[1] - 1) / 2,
                                       (exemplar_img.shape[0] - 1) / 2, self.center_crop_size,
                                       self.center_crop_size)
        instance_img, gt_w, gt_h = self.RandomStretch(instance_img, gt_w, gt_h)
        im_h, im_w, _ = instance_img.shape
        cy_o = (im_h - 1) / 2
        cx_o = (im_w - 1) / 2
        cy = cy_o + np.random.randint(- self.max_translate, self.max_translate + 1)
        cx = cx_o + np.random.randint(- self.max_translate, self.max_translate + 1)
        gt_cx = cx_o - cx
        gt_cy = cy_o - cy
        instance_img, scale = crop_and_pad(instance_img, cx, cy, self.random_crop_size, self.random_crop_size)

        exemplar_img = self.z_transforms(exemplar_img)
        instance_img = self.x_transforms(instance_img)
        # create target
        regression_target, conf_target = self.compute_target(self.anchors,
                                                             np.array(list(map(round, [gt_cx, gt_cy, gt_w, gt_h]))))

        return exemplar_img, instance_img, regression_target, conf_target.astype(np.int64)

    def __len__(self):
        return self.num