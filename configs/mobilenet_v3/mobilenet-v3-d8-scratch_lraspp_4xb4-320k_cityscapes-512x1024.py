_base_ = [
    '../_base_/models/lraspp_m-v3-d8.py', '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# Re-config the data sampler.
model = dict(data_preprocessor=data_preprocessor)
train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=2, num_workers=4)
test_dataloader = val_dataloader

runner = dict(type='IterBasedRunner', max_iters=172000)
