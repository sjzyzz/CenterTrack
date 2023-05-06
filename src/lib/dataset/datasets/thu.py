from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from collections import defaultdict
from ..generic_dataset import GenericDataset


class THU(GenericDataset):
    num_categories = 1
    default_resolution = [512, 512]
    class_name = ['']
    max_objs = 256
    cat_ids = {
        1: 1,
        -1: -1
    }

    def __init__(self, opt, split):
        split = 'trainval' if opt.trainval else 'test'

        data_dir = os.path.join(opt.data_dir, 'thu_basketball_dataset')

        img_dir = os.path.join(data_dir, 'images', '{}'.format(split))

        ann_path = os.path.join(data_dir, 'tracking_annotations', f'{split}.json')

        print(f'annotation path: {ann_path}')

        self.images = None
        # load image list and coco
        super(THU, self).__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)
        print('Loaded THU {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __len__(self):
        return self.num_samples

    def save_tracking_results(self, results, save_dir):
        results_dir = os.path.join(save_dir, 'results_tracking')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        for video in self.coco.dataset['videos']:
            video_id = video['id']
            file_name = video['file_name']
            out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
            f = open(out_path, 'w')
            images = self.video_to_images[video_id]
            tracks = defaultdict(list)
            for image_info in images:
                if not (image_info['id'] in results):
                    continue
                result = results[image_info['id']]
                if not 'frame_id' in image_info:
                    print(image_info)
                frame_id = image_info['frame_id']
                for item in result:
                    if not ('tracking_id' in item):
                        item['tracking_id'] = np.random.randint(100000)
                    if item['active'] == 0:
                        continue
                    tracking_id = item['tracking_id']
                    bbox = item['bbox']
                    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                    tracks[tracking_id].append([frame_id] + bbox)
            rename_track_id = 0
            for track_id in sorted(tracks):
                rename_track_id += 1
                for t in tracks[track_id]:
                    f.write(
                        '{},{},{},{},{},{},1,1,1\n'.format(
                            t[0], rename_track_id, int(t[1]), int(t[2]), int(t[3] - t[1]),
                            int(t[4] - t[2])
                        )
                    )
            f.close()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            if type(all_bboxes[image_id]) != type({}):
                # newest format
                for j in range(len(all_bboxes[image_id])):
                    item = all_bboxes[image_id][j]
                    if item['class'] != 1:
                        continue
                    category_id = 1
                    keypoints = np.concatenate(
                        [
                            np.array(item['hps'], dtype=np.float32).reshape(-1, 2),
                            np.ones((17, 1), dtype=np.float32)
                        ],
                        axis=1
                    ).reshape(51).tolist()
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "score": float("{:.2f}".format(item['score'])),
                        "keypoints": keypoints
                    }
                    if 'bbox' in item:
                        bbox = item['bbox']
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        bbox_out = list(map(self._to_float, bbox[0:4]))
                        detection['bbox'] = bbox_out
                    detections.append(detection)
        return detections

    def save_pose_results(self, results, save_dir):
        json.dump(
            self.convert_eval_format(results),
            open('{}/results_pose.json'.format(save_dir), 'w')
        )

    def run_eval(self, results, save_dir):
        if 'tracking' in self.opt.task:
            self.save_tracking_results(results, save_dir)
            # gt_type_str = '{}'.format(
            #             '_train_half' if '17halftrain' in self.opt.dataset_version \
            #             else '_val_half' if '17halfval' in self.opt.dataset_version \
            #             else '')
            # gt_type_str = '_val_half' if self.year in [16, 19] else gt_type_str
            # gt_type_str = '--gt_type {}'.format(gt_type_str) if gt_type_str != '' else \
            #   ''
            os.system('python tools/eval_motchallenge.py ' + \
                    '../data/thu_basketball_dataset/tracking_annotations/test ' + \
                    '{}/results_tracking/ '.format(save_dir) +' --eval_official')
        if 'multi_pose' in self.opt.task:
            self.save_pose_results(results, save_dir)
            coco_dets = self.coco.loadRes('{}/results_pose.json'.format(save_dir))
            if self.opt.single_class_test:
                for ann in self.coco.dataset['annotations']:
                    ann['category_id'] = 1
            coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            # coco_eval = COCOeval(self.coco, coco_dets, "bbox")
            # coco_eval.evaluate()
            # coco_eval.accumulate()
            # coco_eval.summarize()

    def run_eval_only(self, save_dir):
        # self.save_tracking_results(results, save_dir)
        # gt_type_str = '{}'.format(
        #             '_train_half' if '17halftrain' in self.opt.dataset_version \
        #             else '_val_half' if '17halfval' in self.opt.dataset_version \
        #             else '')
        # gt_type_str = '_val_half' if self.year in [16, 19] else gt_type_str
        # gt_type_str = '--gt_type {}'.format(gt_type_str) if gt_type_str != '' else \
        #   ''
        os.system('python src/tools/eval_motchallenge.py ' + \
                  '/home/sjzyzz/CenterTrack/data/thu_basketball_dataset/tracking_annotations/test ' + \
                  '{}/results_tracking/ '.format(save_dir) +' --eval_official')

        # self.save_pose_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results_pose.json'.format(save_dir))
        if self.opt.single_class_test:
            for ann in self.coco.dataset['annotations']:
                ann['category_id'] = 1
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()