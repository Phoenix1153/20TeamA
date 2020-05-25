from PytorchPlus.active_learning import *
import numpy as np
from torch import nn

X = np.random.random(size=(100, 20))
y = np.random.randint(0, 3, size=(100,))

## GraphDensitySampler 
sampler = GraphDensitySampler(X, y, 0)
already_selected = []
for i in range(3):
    batch = sampler.select_batch_(10, already_selected)
    already_selected.extend(batch)
print('GraphDensitySampler', already_selected)
## test N <= unlabeled datapoint num
batch = sampler.select_batch_(1000, already_selected)
print('N is too big, the batch is', batch)

## kcenter greedy
sampler = kCenterGreedy(X, y, 0)
already_selected = []
for i in range(3):
    batch = sampler.select_batch_(10, already_selected)
    already_selected.extend(batch)
print('kcenter greedy:', already_selected)
## test N <= unlabeled datapoint num
batch = sampler.select_batch_(1000, already_selected)
print('N is too big, the batch is', batch)

## MarginAL
sampler = MarginAL(X, y, 0)
already_selected = []
model = nn.Sequential(
    nn.Linear(20, 3),
    nn.Softmax(dim=-1)
)
for i in range(3):
    batch = sampler.select_batch_(10, already_selected, model)
    already_selected.extend(batch)
print('MarginAL:', already_selected)
## test N <= unlabeled datapoint num
batch = sampler.select_batch_(1000, already_selected, model)
print('N is too big, the batch is', batch)

## informative cluster diverse sampler
sampler = InformativeClusterDiverseSampler(X, y, 0)
already_selected = []
model = nn.Sequential(
    nn.Linear(20, 3),
    nn.Softmax(dim=-1)
)
for i in range(3):
    batch = sampler.select_batch_(10, already_selected, model)
    already_selected.extend(batch)
print('Informative cluster diverse sampler:', already_selected)
## test N <= unlabeled datapoint num
batch = sampler.select_batch_(1000, already_selected, model)
print('N is too big, the batch is', batch)

## represent cluster centers
sampler = RepresentativeClusterMeanSampling(X, y, 0)
already_selected = []
model = nn.Sequential(
    nn.Linear(20, 3),
    nn.Softmax(dim=-1)
)
for i in range(3):
    batch = sampler.select_batch_(10, already_selected, model)
    already_selected.extend(batch)
print('Representative Cluster Mean Sampling:', already_selected)
## test N <= unlabeled datapoint num
batch = sampler.select_batch_(1000, already_selected, model)
print('N is too big, the batch is', batch)

## uniform sampling
sampler = UniformSampling(X, y, 0)
already_selected = []
for i in range(3):
    batch = sampler.select_batch_(10, already_selected)
    already_selected.extend(batch)
print('Uniform sampling:', already_selected)
## test N <= unlabeled datapoint num
batch = sampler.select_batch_(1000, already_selected)
print('N is too big, the batch is', batch)

## mixture samplers
sampler = MixtureOfSamplers(X, y, 0)
already_selected = []
model = nn.Sequential(
    nn.Linear(20, 3),
    nn.Softmax(dim=-1)
)
for i in range(3):
    batch = sampler.select_batch_(10, already_selected, model=model)
    already_selected.extend(batch)
print('mixture sampler:', already_selected)
## test N <= unlabeled datapoint num
batch = sampler.select_batch_(1000, already_selected, model=model)
print('N is too big, the batch is', batch)

## bandit discrete sampler
sampler = BanditDiscreteSampler(X, y, 0)
already_selected = []
model = nn.Sequential(
    nn.Linear(20, 3),
    nn.Softmax(dim=-1)
)
eval_acc = [0.1, 0.2, 0.3, 0.4]
for i in range(3):
    batch = sampler.select_batch_(10, already_selected, eval_acc[i], model=model)
    already_selected.extend(batch)
print('Bandit discrete sampling:', already_selected)
## test N <= unlabeled datapoint num
batch = sampler.select_batch_(1000, already_selected, eval_acc[3], model=model)
print('N is too big, the batch is', batch)

## hierarchical clustering sampler
sampler = HierarchicalClusterAL(X, y, 0)
already_selected = []
for i in range(3):
    batch = sampler.select_batch_(10, already_selected)
    already_selected.extend(batch)
print('Hierarchical clustering sampling:', already_selected)
## test N <= unlabeled datapoint num
batch = sampler.select_batch_(1000, already_selected)
print('N is too big, the batch is', batch)