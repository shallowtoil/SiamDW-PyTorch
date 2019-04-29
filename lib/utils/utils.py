# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

import numpy as np
import json
import glob
import torch
import os

from shapely.geometry import Polygon, box
from os.path import join, realpath, dirname, normpath

# ------------------------
# basic tools
# ------------------------

def load_json(json_path):
    assert(os.path.exists(json_path))
    cfg = json.load(open(json_path, 'r'))
    return cfg


def to_torch(ndarray):
    return torch.from_numpy(ndarray)


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img

# ------------------------
# model tools
# ------------------------

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('missing keys:{}'.format(len(missing_keys)))
    print('unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# ------------------------
# BBOX/RPN tools
# ------------------------

def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def rect_2_cxy_wh(rect):
    return [rect[0]+rect[2]/2, rect[1]+rect[3]/2, rect[2], rect[3]]  # 0-index


def get_min_max_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        w = x2 - x1
        h = y2 - y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h


def judge_overlap(poly, rect):
    xy = poly.reshape(-1, 2)
    polygon_shape = Polygon(xy)
    gridcell_shape = box(rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3])
    # The intersection
    overlap = polygon_shape.intersection(gridcell_shape).area
    return True if overlap > 1 else False

def compute_iou(anchors, box):
    if np.array(anchors).ndim == 1:
        anchors = np.array(anchors)[None, :]
    else:
        anchors = np.array(anchors)
    if np.array(box).ndim == 1:
        box = np.array(box)[None, :]
    else:
        box = np.array(box)
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

    xx1 = np.max([anchor_x1, gt_x1], axis=0)
    xx2 = np.min([anchor_x2, gt_x2], axis=0)
    yy1 = np.max([anchor_y1, gt_y1], axis=0)
    yy2 = np.min([anchor_y2, gt_y2], axis=0)

    inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                           axis=0)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return iou

def nms(bboxes, scores, num, threshold=0.7):
    sort_index = np.argsort(scores)[::-1]
    sort_boxes = bboxes[sort_index]
    selected_bbox = [sort_boxes[0]]
    selected_index = [sort_index[0]]
    for i, bbox in enumerate(sort_boxes):
        iou = compute_iou(selected_bbox, bbox)
        # print(iou, bbox, selected_bbox)
        if np.max(iou) < threshold:
            selected_bbox.append(bbox)
            selected_index.append(sort_index[i])
            if len(selected_bbox) >= num:
                break
    return selected_index

def add_box_img(img, boxes, color=(0, 255, 0)):
    # boxes (x,y,w,h)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    img = img.copy()
    img_ctx = (img.shape[1] - 1) / 2
    img_cty = (img.shape[0] - 1) / 2
    for box in boxes:
        point_1 = [img_ctx - box[2] / 2 + box[0] + 0.5, img_cty - box[3] / 2 + box[1] + 0.5]
        point_2 = [img_ctx + box[2] / 2 + box[0] - 0.5, img_cty + box[3] / 2 + box[1] - 0.5]
        point_1[0] = np.clip(point_1[0], 0, img.shape[1])
        point_2[0] = np.clip(point_2[0], 0, img.shape[1])
        point_1[1] = np.clip(point_1[1], 0, img.shape[0])
        point_2[1] = np.clip(point_2[1], 0, img.shape[0])
        img = cv2.rectangle(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
                            color, 2)
    return img

def box_transform(anchors, gt_box):
    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    gt_cx, gt_cy, gt_w, gt_h = gt_box

    target_x = (gt_cx - anchor_xctr) / anchor_w
    target_y = (gt_cy - anchor_yctr) / anchor_h
    target_w = np.log(gt_w / anchor_w)
    target_h = np.log(gt_h / anchor_h)
    regression_target = np.hstack((target_x, target_y, target_w, target_h))
    return regression_target

# ------------------------
# video tools
# ------------------------

def load_dataset(dataset):
    # buffer controls whether load all images
    info = {}

    if 'OTB' in dataset:
        base_path = join(os.getcwd(), 'dataset', dataset)
        json_path = join(os.getcwd(), 'dataset', dataset, dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            # path_name = info[v]['video_dir']
            info[v]['image_files'] = [join(base_path, im_f) for im_f in info[v]['img_names']]
            info[v]['gt'] = np.array(info[v]['gt_rect'])-[1, 1, 0, 0]
            # we do not use the anno in original files
            info[v]['name'] = v

    elif 'VOT' in dataset:
        base_path = join(os.getcwd(), 'dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            if gt.shape[1] == 4:
                gt = np.column_stack((gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3],
                                      gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3], gt[:, 0] + gt[:, 2], gt[:, 1]))

            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    else:
        print('Not support now, edit for other dataset youself...')
        exit()

    return info

def load_video(video):
    # buffer controls whether load all images
    info = {}
    info[video] = {}

    base_path = normpath(join(realpath(dirname(__file__)), '../dataset', video))

    # ground truth
    gt_path = join(base_path, 'groundtruth_rect.txt')
    gt = np.loadtxt(gt_path, delimiter=',')
    gt = gt - [1, 1, 0, 0]   # OTB for python (if video not from OTB, please delete it)

    # img file name
    img_path = join(base_path, 'img', '*jpg')
    image_files = sorted(glob.glob(img_path))

    # info summary
    info[video]['name'] = video
    info[video]['image_files'] = [join(base_path, im_f) for im_f in img_names]
    info[video]['gt'] = gt

    return info

# ------------------------
# tracking tools
# ------------------------

def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    return im_to_torch(im_patch.copy()) if out_mode in 'torch' else im_patch


def make_scale_pyramid(im, pos, in_side_scaled, out_side, avg_chans):
    in_side_scaled = [round(x) for x in in_side_scaled]
    num_scale = len(in_side_scaled)
    pyramid = torch.zeros(num_scale, 3, out_side, out_side)
    max_target_side = in_side_scaled[-1]
    min_target_side = in_side_scaled[0]
    beta = out_side / min_target_side

    search_side = round(beta * max_target_side)
    search_region = get_subwindow_tracking(im, pos, int(search_side), int(max_target_side), avg_chans, out_mode='np')

    for s, temp in enumerate(in_side_scaled):
        target_side = round(beta * temp)
        pyramid[s, :] = get_subwindow_tracking(search_region, (1 + search_side) / 2, out_side, target_side, avg_chans)

    return pyramid