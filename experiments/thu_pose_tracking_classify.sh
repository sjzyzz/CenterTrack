cd src
# train
python main.py tracking,multi_pose,classify --exp_id thu_pose_track_classify_e140_cw5 --dataset thu --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1,2 --load_model /home/sjzyzz/CenterTrack/models/coco_pose_tracking.pth --trainval --num_epochs 140 --lr_step 90,120 --class_weight 5
# test
python test.py tracking,multi_pose,classify --exp_id thu_pose_track_classify_e140_cw5 --dataset thu --pre_hm --track_thresh 0.4 --pre_thresh 0.5 --resume
cd ..
