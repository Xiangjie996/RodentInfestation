_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/pascal_voc12.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(embed_dims=192, num_heads=3),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=192,
        channels=192,
        num_classes=5,
        num_heads=3,
        embed_dims=192))

optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=172000, val_interval=344)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=8, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=2)
val_dataloader = dict(batch_size=2)
