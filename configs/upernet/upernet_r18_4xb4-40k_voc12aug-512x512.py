_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/pascal_voc12.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained=None,
    backbone=dict(depth=18),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=5),
    auxiliary_head=dict(in_channels=256, num_classes=21))
