import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ignore_class',
        action='store_true',
        help='single class pose estimation or multiple classes pose estimation'
    )
    parser.add_argument('--result_path', type=str, help='the path of reuslt file')
    args = parser.parse_args()

    cocoGt = COCO('data/thu_basketball_dataset/tracking_annotations/test.json')
    resFile = args.result_path
    # new_resFile = f'exp/tracking,multi_pose/{args.exp_id}/K_results.json'
    with open(resFile) as f:
        res_json = json.load(f)
    # 按照score选取每张图片的20的检测框
    annotations_by_img = {}
    annotations = []
    for ann in res_json:
        if not ann['image_id'] in annotations_by_img:
            annotations_by_img[ann['image_id']] = [ann]
        else:
            annotations_by_img[ann['image_id']].append(ann)
    for k, v in annotations_by_img.items():
        K = min(20, len(v))
        # print(v)
        v.sort(key=lambda ele: ele['score'], reverse=True)
        # print(v)
        annotations.extend(v[:K])

    # with open(new_resFile, 'w') as f:
    #     f.write(json.dumps(annotations))

    # if args.select_k:
    #     cocoDt = cocoGt.loadRes(new_resFile)
    # else:
    cocoDt = cocoGt.loadRes(resFile)
    if args.ignore_class:
        for k, v in cocoDt.anns.items():
            v['category_id'] = 1
        for ann in cocoGt.dataset['annotations']:
            ann['category_id'] = 1
    else:
        gt_cls_cnt = [0, 0, 0]
        res_cls_cnt = [0, 0, 0]
        for k, v in cocoDt.anns.items():
            res_cls_cnt[v['category_id']] += 1
        for ann in cocoGt.dataset['annotations']:
            gt_cls_cnt[ann['category_id']] += 1
        print('Actually, there are ', end='')
        for i in range(3):
            print(f'{gt_cls_cnt[i]} instances in class {i}, ', end='')
        print('')
        print('But in PREDICTION, there are ', end='')
        for i in range(3):
            print(f'{res_cls_cnt[i]} instances in class {i}, ', end='')
        print('')
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
