# Adapted from https://raw.githubusercontent.com/nkolot/SPIN/master/datasets/preprocess/coco.py
import os
from os.path import join
import sys
import json
import numpy as np
from pathlib import Path
# from .read_openpose import read_openpose

def coco_extract(dataset_path, out_path):

    # # convert joints to global order
    # joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    # json annotation file
    SPLIT='val'
    json_paths = (Path(dataset_path)/'posetrack_data/annotations'/SPLIT).glob('*.json')
    for json_path in json_paths:
        json_data = json.load(open(json_path, 'r'))

        imgs = {}
        for img in json_data['images']:
            imgs[img['id']] = img

        for annot in json_data['annotations']:
            # keypoints processing
            keypoints = annot['keypoints']
            keypoints = np.reshape(keypoints, (17,3))
            keypoints[keypoints[:,2]>0,2] = 1
            # check if all major body joints are annotated
            if sum(keypoints[5:,2]>0) < 12:
                continue
            # image name
            image_id = annot['image_id']
            img_name = str(imgs[image_id]['file_name'])
            # img_name_full = f'images/{SPLIT}/{json_path.stem}/{img_name}'
            img_name_full = img_name

            # keypoints
            part = np.zeros([17,3])
            # part[joints_idx] = keypoints
            part = keypoints

            # scale and center
            bbox = annot['bbox']
            center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
            # scale = scaleFactor*max(bbox[2], bbox[3]) # Don't do /200
            scale = scaleFactor*np.array([bbox[2], bbox[3]]) # Don't /200
            # # read openpose detections
            # json_file = os.path.join(openpose_path, 'coco',
            #     img_name.replace('.jpg', '_keypoints.json'))
            # openpose = read_openpose(json_file, part, 'coco')

            # store data
            imgnames_.append(img_name_full)
            centers_.append(center)
            scales_.append(scale)
            parts_.append(part)
            # openposes_.append(openpose)


    # NOTE: Posetrack val doesn't annotate ears (17,18)
    # But Posetrack does annotate head, neck so that wil have to live in extra_kps.
    posetrack_to_op_extra = [0, 37, 38, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11] # Will contain 15 keypoints.
    all_keypoints_2d = np.zeros((len(parts_), 44, 3))
    all_keypoints_2d[:,posetrack_to_op_extra] = np.stack(parts_)[:,:len(posetrack_to_op_extra),:3]
    body_keypoints_2d = all_keypoints_2d[:,:25,:]
    extra_keypoints_2d = all_keypoints_2d[:,25:,:]

    print(f'{extra_keypoints_2d.shape=}')

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, f'posetrack_2018_{SPLIT}.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       body_keypoints_2d=body_keypoints_2d,
                       extra_keypoints_2d=extra_keypoints_2d,
                    #    openpose=openposes_
            )

if __name__ == '__main__':
    coco_extract('/shared/pavlakos/datasets/posetrack/posetrack2018/', 'hmr2_evaluation_data/')
