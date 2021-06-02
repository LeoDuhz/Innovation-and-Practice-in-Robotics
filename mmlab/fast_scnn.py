_base_ = [
    './configs/fastscnn.py', './configs/linemod_dataset.py',
    './configs/runtime.py', './configs/schedule.py'
]

# Re-config the data sampler.
data = dict(samples_per_gpu=2, workers_per_gpu=4)

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)
