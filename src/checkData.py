import os
import torch
fileName = 'C:/MachineLearning/CLsurveyICLdevelopCopy/srclCLcopy/data/datasets/tiny-imagenet/tiny-imagenet-200/no_crop/5tasks/1/imgfolder_trainvaltest_rndtrans.pth.tar'

dsets = torch.load(fileName, map_location='cpu')
print(dsets)

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4,
                                                   shuffle=True, num_workers=0, pin_memory=True)
                    for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes
print(dset_sizes, dset_classes)
print(dsets['val'])
count = 0
for data in dsets['val']:
    # get the inputs

    # inputs, labels = data
    print(count, data)
    count += 1
# count = 0
# for data in dset_loaders['val']:
#     # get the inputs
#
#     inputs, labels = data
#     print(count, inputs, labels)
#     count += 1