'''在CenterTrack目录下使用此脚本'''
import json
import os

DATA_PATH = 'data/thu_basketball_dataset/annotations'
SPLIT = ['trainval', 'test']
OUT_PATH = 'data/thu_basketball_dataset/tracking_annotations'

if __name__ == '__main__':
    for split in SPLIT:
        out_path = os.path.join(OUT_PATH, f'{split}.json')
        out = {
            'images': [],
            'annotations': [],
            "categories":
                [
                    {
                        "supercategory": "",
                        "id": 0,
                        "name": "offense player"
                    }, {
                        "supercategory": "",
                        "id": 1,
                        "name": "defense player"
                    }, {
                        "supercategory": "",
                        "id": 2,
                        "name": "referee"
                    }
                ],
            'videos': []
        }
        with open(os.path.join(DATA_PATH, f'{split}.json')) as f:
            json_data = json.load(f)
        video_dict = {}
        for image in json_data['images']:
            image_id = image['id']
            video_id = int(image_id / 1000)
            frame_id = int(image_id % 1000)
            if video_id not in video_dict:
                video_dict[video_id] = []
            video_dict[video_id].append({
                'frame_id': frame_id,
                'image': image
            })
        for k, v in video_dict.items():
            v.sort(key=lambda e: e['frame_id'])

        video_cnt = 0
        image_dict = {}
        for k, v in video_dict.items():
            video_cnt += 1
            out['videos'].append({
                'id': video_cnt,
                'file_name': str(k)
            })
            for i in range(len(v)):
                image_info = {
                    'file_name': v[i]['image']['file_name'],
                    'id': v[i]['image']['id'],
                    'frame_id': i + 1,
                    'prev_image_id': v[i - 1]['image']['id'] if i > 0 else -1,
                    'next_image_id': v[i + 1]['image']['id'] if i < len(v) - 1 else -1,
                    'video_id': video_cnt
                }
                out['images'].append(image_info)
                image_dict[image_info['id']] = image_info

        anns = json_data['annotations']
        ann_cnt = 0
        for i in range(len(anns)):
            ann = anns[i]
            image = image_dict[ann['image_id']]
            frame_id = int(image['frame_id'])
            track_id = int(ann['track_id'])
            cat_id = int(ann['category_id'])
            ann_cnt += 1
            ann = {
                'id': ann_cnt,
                'category_id': cat_id,
                'image_id': ann['image_id'],
                'track_id': track_id,
                'bbox': ann['bbox'],
                'keypoints': ann['keypoints'],
                'num_keypoints': ann['num_keypoints'],
                'area': ann['area'],
                'conf': 1,
                'iscrowd': ann['iscrowd'],
            }
            out['annotations'].append(ann)

        json.dump(out, open(out_path, 'w'))
