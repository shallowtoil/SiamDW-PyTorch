# -*- coding:utf-8 -*-
# Licensed under The MIT License

import argparse
import numpy as np
import cv2
import os
import random
import torch
import json
import sys

sys.path.append(os.getcwd())
import lib.net.models as models
from torch.autograd import Variable
from lib.utils.utils import load_pretrain, load_json, to_torch, im_to_torch, \
    load_dataset, load_video, get_subwindow_tracking, make_scale_pyramid, \
    cxy_wh_2_rect, get_min_max_bbox, judge_overlap
from lib.utils.config import configSiamRPN as p
from lib.dataset.generate_target import generate_anchors

parser = argparse.ArgumentParser(description='PyTorch Tracking Test')
parser.add_argument('--arch', dest='arch', default='SiamRPN_Res22', help='architecture of pretrained model')
parser.add_argument('--resume', default='./models/CIResNet22.pth', type=str, help='pretrained model')
parser.add_argument('--dataset', default='OTB2015', choices=['OTB2015', 'OTB2013', 'VOT2017', 'none'],
                    help='dataset test')
parser.add_argument('--video', default='', help='dataset test')
parser.add_argument('--vis', default=False, help='whether to visualize result')

def init(im, target_pos, target_sz, model):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    p = RPNConfig()

    net = model
    p.anchor = generate_anchors(p.total_stride, p.scales, p.ratios, p.score_size)

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = python2round(np.sqrt(wc_z * hc_z))

    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.template(z.cuda())

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.expand_dims(window, axis=0)  # [1,17,17]
    window = np.repeat(window, p.anchor_num, axis=0)  # [5,17,17]

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz

    return state

def update(net, x_crop, target_pos, target_sz, window, scale_z, p):
    score, delta = net.track(x_crop)

    b, c, s, s = delta.size()
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1, s, s).data.cpu().numpy()  # [4,5,17,17]
    score = torch.sigmoid(score).squeeze().cpu().data.numpy()  # [5,17,17]

    delta[0, ...] = delta[0, ...] * p.anchor[2, ...] + p.anchor[0, ...]
    delta[1, ...] = delta[1, ...] * p.anchor[3, ...] + p.anchor[1, ...]
    delta[2, ...] = np.exp(delta[2, ...]) * p.anchor[2, ...]
    delta[3, ...] = np.exp(delta[3, ...]) * p.anchor[3, ...]

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, ...], delta[3, ...]) / (sz_wh(target_sz)))  # scale penalty  [5,17,17]
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, ...] / delta[3, ...]))  # ratio penalty  [5,17,17]

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)  # [5,17,17]
    pscore = penalty * score  # [5, 17, 17]

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence  # [5, 17, 17]
    a_max, r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

    target = delta[:, a_max, r_max, c_max] / scale_z  # [4,1]

    target_sz = target_sz / scale_z
    lr = penalty[a_max, r_max, c_max] * score[a_max, r_max, c_max] * p.lr  # lr for OTB

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    return target_pos, target_sz, score[a_max, r_max, c_max]

def track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(
        get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans).unsqueeze(0))

    target_pos, target_sz, score = update(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state


def track_video(model, video):
    start_frame, toc = 0, 0

    # vis or save OTB result to evaluate
    if not args.vis:
        tracker_path = os.path.join('test', args.dataset,
                                    args.arch.split('.')[0] + args.resume.split('/')[-1].split('.')[0])

        if not os.path.exists(tracker_path):
            os.makedirs(tracker_path)

        if 'VOT' in args.dataset:
            baseline_path = os.path.join(tracker_path, 'baseline')
            video_path = os.path.join(baseline_path, video['name'])
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, video['name'] + '_001.txt')
        else:
            result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

        if not os.path.exists(result_path):  # for multi-gpu test
            fin = open(result_path, "w")
            fin.close()
        else:
            return

    regions = []
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        if not os.path.isfile(image_file):
            print('NOT existed', image_file)
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        tic = cv2.getTickCount()

        if f == start_frame:  # init
            cx, cy, w, h = get_min_max_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = SiamFC_init(im, target_pos, target_sz, model)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:  # tracking
            state = SiamFC_track(state, im)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = judge_overlap(gt[f], location) if 'VOT' in args.dataset else 1
            if b_overlap:
                regions.append(location)
            else:
                regions.append(2)
                start_frame = f + 5
        else:
            regions.append(0)

        toc += cv2.getTickCount() - tic

    if bool(args.vis) and f >= start_frame:  # visualization (skip lost frame)
        if f == 0:
            cv2.destroyAllWindows()
            cv2.rectangle(im, (int(gt[f, 0]), int(gt[f, 1])), (int(gt[f, 0] + gt[f, 2]), int(gt[f, 1] + gt[f, 3])),
                          (0, 255, 0), 3)
        else:
            location = [int(l) for l in location]  #
            cv2.rectangle(im, (location[0], location[1]), (location[0] + location[2], location[1] + location[3]),
                          (0, 255, 255), 3)
        cv2.putText(im, '#' + str(f), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(video['name'], im)
        cv2.waitKey(1)

    else:
        with open(result_path, "w") as fin:
            if 'VOT' in args.dataset:
                for x in regions:
                    if isinstance(x, int):
                        fin.write("{:d}\n".format(x))
                    else:
                        p_bbox = x.copy()
                        if p_bbox[0] < 0: p_bbox[0] = 0
                        if p_bbox[1] < 0: p_bbox[1] = 0
                        fin.write(','.join([str(i) for i in p_bbox]) + '\n')
            else:
                for x in regions:
                    p_bbox = x.copy()
                    if p_bbox[0] < 0: p_bbox[0] = 1
                    if p_bbox[1] < 0: p_bbox[1] = 1
                    fin.write(','.join(
                        [str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))


def main():
    global args, total_lost
    total_lost = 0
    args = parser.parse_args()
    model = models.__dict__[args.arch]()

    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model = load_pretrain(model, args.resume)

    model.eval()
    model = model.cuda()

    if args.video and not args.dataset == 'none':
        dataset = load_video(args.video)
        track_video(model, dataset[args.video])
    else:
        dataset = load_dataset(args.dataset)
        video_keys = list(dataset.keys()).copy()
        random.shuffle(video_keys)
        for video in video_keys:
            track_video(model, dataset[video])

if __name__ == '__main__':
    main()

