cd src
# train
# python main.py multi_pose --exp_id thu_pose_track_e140_without_track --dataset thu --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1,2 --load_model /home/sjzyzz/CenterTrack/models/coco_pose_tracking.pth --trainval --num_epochs 140 --lr_step 90,120
# test
python test.py multi_pose,classify --exp_id thu_pose_track_e140_without_track --dataset thu --pre_hm --track_thresh 0.4 --pre_thresh 0.5 --resume --single_class_test
cd ..
