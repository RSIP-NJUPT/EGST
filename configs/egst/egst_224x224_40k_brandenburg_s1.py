_base_ = [
    "../_base_/models/egst.py",
    "../_base_/datasets/brandenburg_s1_224x224.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
num_classes = 15
# model settings
class_weight = None
model = dict(
    backbone=dict(
        # EGSTNet 和 GraphTransformer 一样（仅有部分模块名称不同）
        # 权重是基于 GraphTransformer 的训练的
        # type="EGSTNet", 
        type="GraphTransformer", # 加载预训练权重用 GraphTransformer
        in_chans=2,
        num_classes=num_classes,
        embed_dims=[120, 240, 360, 480],
        window_size=[3, 7, 7],
        depths=[1, 1, 2, 1],
        dataset_flag="brandenburg", # todo
        nodes_keep_ratio=0.1,
        RF=23, # 7 11 23 35 41 53
    ),
    decode_head=dict(
        type="PseudoHead",
        num_classes=num_classes,
        ignore_index=255,  # default
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            use_focal_loss=False,  # use fl
            use_fl_cel=False,  # use fl and cel
            loss_weight=1.0,
            class_weight=class_weight,
        ),
    ),
)

# test_cfg = dict(mode='slide', crop_size=(224, 224), stride=(200, 200))