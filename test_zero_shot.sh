

### test on the MVTec AD dataset


python vl_test.py --mode zero_shot --dataset mvtec \
--data_path ./data/mvtec --save_path ./results/mvtec/zero_shot_vis \
--config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--model ViT-B-16-plus-240 --features_list 3 6 9 12 --pretrained laion400m_e32 --image_size 240 --seed 111 --adapter True --epoch 5

### test on the VisA dataset

python vis_test.py --mode zero_shot --dataset visa \
--data_path ./data/visa --save_path ./results/visa/zero_shot_clip \
--config_path ./open_clip/model_configs/ViT-B-16-plus-240.json \
--model ViT-B-16-plus-240 --features_list 3 6 9 12 --pretrained laion400m_e32 --image_size 240 --seed 42 --adapter True --epoch 5