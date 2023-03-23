# RUN THIS AS INIT
from data.dataset import *

from models.net import *
from utilities.main_postprocessing import *

# CONFIG
config = utils.get_parsed_config()
test_results_root_path = utils.read_from_config(config, 'test_results_root_path')
tr_results_root_path = utils.read_from_config(config, 'tr_results_root_path')
models_root_path = utils.read_from_config(config, 'models_root_path')

dataset = TinyImgnetDataset()
model = SmallVGG9(models_root_path, dataset.input_size)

# Turn on/off
plot_SI = True

# PARAMS
img_extention = 'png'  # 'eps' for latex
save_img = True

plot_seq_acc = True
plot_seq_forgetting = False
hyperparams_selection = []

label_segment_idxs = [0]
exp_name_contains = None

# INIT
method_names = []
method_data_entries = []
# gridsearch_names = ['optimizer_Adam', 'optimizer_Adam_repeat2', 'optimizer_Adam_repeat3', 'optimizer_Adam_repeat4', 'optimizer_Adam_repeat5']
# gridsearch_names = ['optimizer_Adam_CCL', 'optimizer_Adam_CCL_repeat2', 'optimizer_Adam_CCL_repeat3', 'optimizer_Adam_CCL_repeat4', 'optimizer_Adam_CCL_repeat5']
# gridsearch_names = ['optimizer_Adam_CCL_lambda10', 'optimizer_Adam_CCL_lambda10_repeat2', 'optimizer_Adam_CCL_lambda10_repeat3', 'optimizer_Adam_CCL_lambda10_repeat4', 'optimizer_Adam_CCL_lambda1_repeat5']
# gridsearch_names = ['optimizer_Adam_CCL_2iters', 'optimizer_Adam_CCL_2iters_repeat2', 'optimizer_Adam_CCL_2iters_repeat3', 'optimizer_Adam_CCL_2iters_repeat4', 'optimizer_Adam_CCL_2iters_repeat5']
# gridsearch_names = ['optimizer_Adam_STCL_redo', 'optimizer_Adam_STCL_redo_repeat2', 'optimizer_Adam_STCL_redo_repeat3', 'optimizer_Adam_STCL_redo_repeat4', 'optimizer_Adam_STCL_redo_repeat5']
# gridsearch_names = ['optimizer_Adam_STCL_redo_lambda1', 'optimizer_Adam_STCL_redo_lambda1_repeat2', 'optimizer_Adam_STCL_redo_lambda1_repeat3', 'optimizer_Adam_STCL_redo_lambda1_repeat4', 'optimizer_Adam_STCL_redo_lambda1_repeat5']
# gridsearch_names = ['SGD_FT_lambda1', 'SGD_FT_lambda1_repeat2', 'SGD_FT_lambda1_repeat3', 'SGD_FT_lambda1_repeat4', 'SGD_FT_lambda1_repeat5']
# gridsearch_names = ['SGD_FT_correct', 'SGD_FT_correct_repeat2', 'SGD_FT_correct_repeat3', 'SGD_FT_correct_repeat4', 'SGD_FT_correct_repeat5']
# gridsearch_names = ['Adam_FT_renewOpt2', 'Adam_FT_renewOpt2_repeat2', 'Adam_FT_renewOpt2_repeat3', 'Adam_FT_renewOpt2_repeat4', 'Adam_FT_renewOpt2_repeat5']
# gridsearch_names = ['Adam_FT_unbalancedData', 'Adam_FT_unbalancedData_repeat2', 'Adam_FT_unbalancedData_repeat3', 'Adam_FT_unbalancedData_repeat4', 'Adam_FT_unbalancedData_repeat5']
# gridsearch_names = ['SGD_unbalancedData', 'SGD_unbalancedData_repeat2', 'SGD_unbalancedData_repeat3', 'SGD_unbalancedData_repeat4', 'SGD_unbalancedData_repeat5']
# gridsearch_names = ['Adam_FT_unbalancedData_long', 'Adam_FT_unbalancedData_long_repeat2', 'Adam_FT_unbalancedData_long_repeat3', 'Adam_FT_unbalancedData_long_repeat4', 'Adam_FT_unbalancedData_long_repeat5']
# gridsearch_names = ['Adam_FT_unbalancedDataAmount', 'Adam_FT_unbalancedDataAmount_repeat2', 'Adam_FT_unbalancedDataAmount_repeat3', 'Adam_FT_unbalancedDataAmount_repeat4', 'Adam_FT_unbalancedDataAmount_repeat5']
# gridsearch_names = ['Adam_FT_unbalancedDataAmountMid', 'Adam_FT_unbalancedDataAmountMid_repeat2', 'Adam_FT_unbalancedDataAmountMid_repeat3', 'Adam_FT_unbalancedDataAmountMid_repeat4', 'Adam_FT_unbalancedDataAmountMid_repeat5']
# gridsearch_names = ['Adam_unbalancedDataAmountMidLastEpoch', 'Adam_unbalancedDataAmountMidLastEpoch_repeat2', 'Adam_unbalancedDataAmountMidLastEpoch_repeat3', 'Adam_unbalancedDataAmountMidLastEpoch_repeat4', 'Adam_unbalancedDataAmountMidLastEpoch_repeat5']
# gridsearch_names = ['Adam_unbalancedDataAmountLastEpoch', 'Adam_unbalancedDataAmountLastEpoch_repeat2', 'Adam_unbalancedDataAmountLastEpoch_repeat3', 'Adam_unbalancedDataAmountLastEpoch_repeat4', 'Adam_unbalancedDataAmountLastEpoch_repeat5']
# gridsearch_names = ['Adam_unbalancedDataAmountLastEpochE100', 'Adam_unbalancedDataAmountLastEpochE100_repeat2', 'Adam_unbalancedDataAmountLastEpochE100_repeat3', 'Adam_unbalancedDataAmountLastEpochE100_repeat4', 'Adam_unbalancedDataAmountLastEpochE100_repeat5']
# gridsearch_names = ['Adam_10Classes_noisy25_ICL_reg_repeat', 'Adam_10Classes_noisy25_ICL_reg_repeat2', 'Adam_10Classes_noisy25_ICL_reg_repeat3', 'Adam_10Classes_noisy25_ICL_reg_repeat4']
# gridsearch_names = ['Adam_10Classes_noisy25_ICL_5ep_repeat', 'Adam_10Classes_noisy25_ICL_5ep_repeat2', 'Adam_10Classes_noisy25_ICL_5ep_repeat3', 'Adam_10Classes_noisy25_ICL_5ep_repeat4', 'Adam_10Classes_noisy25_ICL_5ep_repeat5']

 #['base_training_adam_ReloadOp_SH_reweight', 'base_training_adam_ReloadOp_SH_reweight_repeat2',  'base_training_adam_ReloadOp_SH_reweight_repeat3','base_training_adam_ReloadOp_SH_reweight_repeat4']
# gridsearch_names = ['base_training_adam_ReloadOp_SH_reweight_repeat5',
#                     'base_training_adam_ReloadOp_SH_reweight_repeat6', 'base_training_adam_ReloadOp_SH_reweight_repeat12', 'base_training_adam_ReloadOp_SH_reweight_repeat13', 'base_training_adam_ReloadOp_SH_reweight_repeat14', 'base_training_adam_ReloadOp_SH_reweight_repeat15','base_training_adam_ReloadOp_SH_reweight_repeat16',
#                     'base_training_adam_ReloadOp_SH_reweight_repeat7', 'base_training_adam_ReloadOp_SH_reweight_repeat8', 'base_training_adam_ReloadOp_SH_reweight_repeat9', 'base_training_adam_ReloadOp_SH_reweight_repeat10', 'base_training_adam_ReloadOp_SH_reweight_repeat11',
# 'base_training_adam_ReloadOp_SH_reweight_repeat30',
# 'base_training_adam_ReloadOp_SH_reweight_repeat31',
#                     ]
# gridsearch_names = ['base_training', 'base_training_reloadOp', 'base_training_reloadOp_SHRedo', 'base_training_paper', 'base_training_reloadOp_paper', 'base_training_reloadOp_SH_paper', 'base_training_Adam_paper', 'base_training_reloadOp_Adam_paper', 'base_training_reloadOp_Adam_paper_redo', 'base_training_reloadOp_SH_Adam_paper']
# multi_head_list = [True, True, False, True, True, False, True,  True, True, False]
# legends = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6', 'FT7', 'FT8', 'FT9', 'FT10']
gridsearch_names = ['base_training', 'base_training_reloadOp', 'base_training_renewOp_SH', 'base_training_reloadOp_SHRedo',  'base_training_Adam_paper', 'base_training_reloadOp_Adam_paper', 'base_training_renewOp_SH_Adam_paper', 'base_training_reloadOp_SH_Adam_paper']
multi_head_list = [True, True, False, False, True, True, False, False]
# legends = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6']
legends = ['SGD+NOp+MH', 'SGD+ROp+MH', 'SGD+NOp+SH', 'SGD+ROp+SH', 'Adam+NOp+MH', 'Adam+ROp+MH', 'Adam+NOp+SH', 'Adam+ROp+SH']
#############################################
methods = [Joint(), SI(), EWC(), LWF(), EBLL()]
# for method in methods:
for gridsearch_name in gridsearch_names:
    # method = IMM('mean')
    # method = IMM('mode')
    # method = SI()
    # method = MAS()
    # method = LWF()
    # method = Joint()
    method = FineTuning()
    method_names.append(method.name)
    label = None

    tuning_selection = []

    method_data_entries.extend(
        collect_gridsearch_exp_entries(test_results_root_path, tr_results_root_path, dataset, method, gridsearch_name,
                                       model, tuning_selection, label_segment_idxs=label_segment_idxs,
                                       exp_name_contains=exp_name_contains))

#############################################
# ANALYZE
#############################################
print(method_data_entries)
out_name = None
if save_img:
    out_name = '_'.join(['DEMO', dataset.name, "(" + '_'.join(method_names) + ")", model.name])

analyze_experiments_icl(method_data_entries, hyperparams_selection=hyperparams_selection, plot_seq_acc=plot_seq_acc,
                    plot_seq_forgetting=plot_seq_forgetting, save_img_parent_dir=out_name, all_diff_color_force=True,
                        img_extention=img_extention, taskcount=5, n_iters=1, gridsearch_name=gridsearch_name,
                        multi_head=multi_head_list, legends=legends)
