import os
import torch
import numpy as np
fileName = 'C:/MachineLearning/CLsurveyMCofficial/src/results/train/tiny_imgnet/IMM/small_VGG9_cl_128_128/gridsearch/AdamSWT_N5/dm=0.2_df=0.5_e=50_bs=100_lambda=0.01/task_3/TASK_TRAINING/best_model.pth.tar'
fileName2 = 'C:/MachineLearning/CLsurveyMCofficial/src/results/train/tiny_imgnet/IMM/small_VGG9_cl_128_128/gridsearch/AdamSWT_N5/dm=0.2_df=0.5_e=50_bs=100_lambda=0.01/task_3/TASK_TRAINING/best_model_mean_merge.pth.tar'
# fileName2 = 'C:/MachineLearning/CLsurveyMCofficial/src/results/train/tiny_imgnet/IMM/small_VGG9_cl_128_128/gridsearch/AdamSWT_N5/dm=0.2_df=0.5_e=50_bs=100_lambda=0.01/task_2/TASK_TRAINING/best_model_mode_merge.pth.tar'

model = torch.load(fileName, map_location='cpu')
mean_model = torch.load(fileName2, map_location='cpu')
state_dict = model.state_dict()
mean_state_dict = mean_model.state_dict()
for param_name, param_value in model.named_parameters():
    param_value_mean = mean_state_dict[param_name]
    print(param_name, np.sum(param_value.detach().numpy()), np.sum(param_value.detach().numpy() - param_value_mean.detach().numpy()))
#
# dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4,
#                                                    shuffle=True, num_workers=0, pin_memory=True)
#                     for x in ['train', 'val']}
# dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
# dset_classes = dsets['train'].classes
# print(dset_sizes, dset_classes)
# print(dsets['val'])
# count = 0
# for data in dsets['val']:
#     # get the inputs
#
#     # inputs, labels = data
#     print(count, data)
#     count += 1
# # count = 0
# # for data in dset_loaders['val']:
# #     # get the inputs
# #
# #     inputs, labels = data
# #     print(count, inputs, labels)
# #     count += 1