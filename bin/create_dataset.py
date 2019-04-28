import numpy as np
import pickle
import os
import cv2
import functools
import xml.etree.ElementTree as ET
import sys
sys.path.append(os.getcwd())

from multiprocessing import Pool
from fire import Fire
from tqdm import tqdm
from glob import glob

from siamfc import config, get_instance_image, xyxy2cxcywh

def worker(output_dir, video_dir):
    image_names = glob(os.path.join(video_dir, '*.JPEG'))
    image_names = sorted(image_names,
                        key=lambda x:int(x.split('/')[-1].split('.')[0]))
    video_name = video_dir.split('/')[-1]
    save_folder = os.path.join(output_dir, video_name)
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)
    trajs = {}
    wh_info = []
    for image_name in image_names:
        # img = cv2.imread(image_name)
        # img_mean = tuple(map(int, img.mean(axis=(0, 1))))

        anno_name = image_name.replace('Data', 'Annotations')
        anno_name = anno_name.replace('JPEG', 'xml')
        tree = ET.parse(anno_name)
        root = tree.getroot()
        bboxes = []
        filename = root.find('filename').text
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))
            trkid = int(obj.find('trackid').text)
            if trkid in trajs:
                trajs[trkid].append(filename)
            else:
                trajs[trkid] = [filename]
            # instance_img, w, h = get_instance_image(img, bbox,
            #         config.exemplar_size, config.instance_size, config.context_amount, img_mean)
            cx, cy, w, h = xyxy2cxcywh(bbox)
            wc_z = w + config.context_amount * (w + h)
            hc_z = h + config.context_amount * (w + h)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = config.exemplar_size / s_z
            d_search = (config.instance_size - config.instance_size) / 2
            pad = d_search / scale_z
            s_x = s_z + 2 * pad
            scale_x = config.instance_size / s_x
            w = round((w*scale_x),2)
            h = round((h*scale_x),2)
            instance_img_name = os.path.join(save_folder, filename+".{:02d}.x.jpg".format(trkid))
            wh_info.append([instance_img_name, w, h])
            # cv2.imwrite(instance_img_name, instance_img)
    return wh_info

def processing(data_dir, output_dir, num_threads=8):
    # get all 4417 videos
    video_dir = os.path.join(data_dir, 'Data/VID')
    all_videos = glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
                 glob(os.path.join(video_dir, 'val/*'))
    # meta_data = []
    wh_data = []
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(
            functools.partial(worker, output_dir), all_videos), total=len(all_videos)):
            wh_data.append(ret)

    # save meta data
    pickle.dump(wh_data, open(os.path.join(output_dir, "wh_data.pkl"), 'wb'))


if __name__ == '__main__':
    Fire(processing)

