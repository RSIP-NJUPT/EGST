# train
CUDA_VISIBLE_DEVICES=3 PORT=13579 tools/dist_train.sh configs/egst/egst_224x224_40k_slovenia.py 1
CUDA_VISIBLE_DEVICES=2 PORT=13571 tools/dist_train.sh configs/egst/egst_224x224_40k_brandenburg_s1.py 1
CUDA_VISIBLE_DEVICES=1 PORT=13572 tools/dist_train.sh configs/egst/egst_224x224_40k_brandenburg_s2.py 1

# test
CUDA_VISIBLE_DEVICES=3 PORT=13579 tools/dist_test.sh configs/egst/egst_224x224_40k_slovenia.py /opt/data/private/SL-GT2/work_dirs/gt_224x224_40k_slovenia/train_20241025_022920/iter_40000.pth 1

CUDA_VISIBLE_DEVICES=0 PORT=13572 tools/dist_test.sh configs/egst/egst_224x224_40k_brandenburg_s1.py /opt/data/private/SL-GT2/work_dirs/gt_224x224_40k_brandenburg_s1/train_20241023_200258/iter_36000.pth 1

CUDA_VISIBLE_DEVICES=1 PORT=13571 tools/dist_test.sh configs/egst/egst_224x224_40k_brandenburg_s2.py /opt/data/private/SL-GT2/work_dirs/gt_224x224_40k_brandenburg_s2/train_20241024_010013/iter_40000.pth 1
CUDA_VISIBLE_DEVICES=1 PORT=13571 tools/dist_test.sh configs/egst/egst_224x224_40k_brandenburg_s2.py /opt/data/private/SL-GT2/work_dirs/aslka_pisgsa_224x224_40k_brandenburg_s2/train_20241125_042448/iter_40000.pth 1
