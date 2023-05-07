from collections import defaultdict
import json
import os
import os.path as osp

import numpy as np

if __name__ == '__main__':
    for split in ['test']:
        json_path = f'data/thu_basketball_dataset/tracking_annotations/{split}.json'
        with open(json_path) as f:
            json_data = json.load(f)
        video_to_images = {}
        for image in json_data['images']:
            video_id = image['video_id']
            if not video_id in video_to_images:
                video_to_images[video_id] = []
            video_to_images[video_id].append(image)
        results = {}  # 将图片id映射到对应的ann
        for ann in json_data['annotations']:
            image_id = ann['image_id']
            if not image_id in results:
                results[image_id] = []
            results[image_id].append(ann)

        create_location = f'data/thu_basketball_dataset/tracking_annotations/{split}'
        if not os.path.exists(create_location):
            os.mkdir(create_location)
        for video in json_data['videos']:
            video_id = video['id']
            file_name = video['file_name']
            folder_path = osp.join(create_location, file_name)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            gt_folder_path = osp.join(folder_path, 'gt')
            if not os.path.exists(gt_folder_path):
                os.mkdir(gt_folder_path)
            out_path = osp.join(gt_folder_path, '{}.txt'.format(file_name))
            with open(out_path, 'w') as f:
                images = video_to_images[video_id]
                tracks = defaultdict(list)
                for image_info in images:
                    result = results[image_info['id']]
                    frame_id = image_info['frame_id']
                    for item in result:
                        if not ('tracking_id' in item):
                            item['tracking_id'] = np.random.randint(100000)
                            print('There is something wrong!!')
                        # if item['active'] == 0:
                        #     continue
                        tracking_id = item['tracking_id']
                        bbox = item['bbox']
                        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                        tracks[tracking_id].append(
                            [frame_id] + bbox + [item['category_id']]
                        )
                    rename_track_id = 0
                for track_id in sorted(tracks):
                    # TODO 最后三位需要考虑
                    rename_track_id += 1
                    for t in tracks[track_id]:
                        f.write(
                            '{},{},{:d},{:d},{:d},{:d},1,{:d},1\n'.format(
                                t[0], rename_track_id, int(t[1]), int(t[2]), int(t[3]),
                                int(t[4]), int(t[5])
                            )
                        )