# model settings

model = dict(
    type="EncoderDecoder",
    pretrained=None,
    backbone=dict(
        # EGSTNet 和 GraphTransformer 一样（仅有部分模块名称不同）
        # 权重是基于 GraphTransformer 的训练的
        # type="EGSTNet", 
        type="GraphTransformer", # 加载预训练权重用 GraphTransformer
        in_chans=2,
        num_classes=8,
        embed_dims=[120, 240, 360, 480],
        window_size=[2, 7, 7],
        depths=[1, 1, 2, 1],
        dataset_flag="slovenia",
        nodes_keep_ratio=0.5
    ),
    decode_head=dict(
        type="PseudoHead", 
        num_classes=8, 
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)
    ),
)

# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode="whole")
