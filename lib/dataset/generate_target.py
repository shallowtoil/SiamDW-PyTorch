# ------------------------------------------------------------------------------
# Written by Jinghao Zhou
# Email: jensen.zhoujh@qq.com
# Details: generate prior anchors for RPN-based trackers
# Date: 2019.4.28
# ------------------------------------------------------------------------------

import numpy as np

def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = base_size * base_size
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    # (5,4x225) to (225x5,4)
    ori = - (score_size // 2) * total_stride
    # the left displacement
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    # (15,15)
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    # (15,15) to (225,1) to (5,225) to (225x5,1)
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

def generate_scoremap(score_size):
    h, w = score_size
    y = np.arange(h, dtype=np.float32) - (h - 1) / 2.
    x = np.arange(w, dtype=np.float32) - (w - 1) / 2.
    y, x = np.meshgrid(y, x)
    dist = np.sqrt(x ** 2 + y ** 2)
    mask = np.zeros((h, w))
    mask[dist <= config.radius / config.total_stride] = 1
    mask = mask[np.newaxis, :, :]
    weights = np.ones_like(mask)
    weights[mask == 1] = 0.5 / np.sum(mask == 1)
    weights[mask == 0] = 0.5 / np.sum(mask == 0)
    mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
    return mask.astype(np.float32), weights.astype(np.float32)