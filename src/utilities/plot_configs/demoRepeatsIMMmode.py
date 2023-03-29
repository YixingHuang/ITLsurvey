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

# gridsearch_names = ['AdamSWT_N5_run1', 'AdamSWT_N5_run2',
#                     'AdamSWT_N5_run3', 'AdamSWT_N5_run4', 'AdamSWT_N5_run5',
#                    'AdamSWT_N5_run6',
#                     'AdamSWT_N5_run7', 'AdamSWT_N5_run8',
#                     'AdamSWT_N5_run9', 'AdamSWT_N5_run10']
# gridsearch_names = ['AdamSWT_N3','AdamSWT_N3_repeat1', 'AdamSWT_N3_repeat2',
#                     'AdamSWT_N3_repeat3', 'AdamSWT_N3_repeat4', 'AdamSWT_N3_repeat5',
#                     'AdamSWT_N3_repeat6',  'AdamSWT_N3_repeat7', 'AdamSWT_N3_repeat8',
#                     'AdamSWT_N3_repeat9', 'AdamSWT_N3_repeat10']
gridsearch_names = ['AdamSWT_N3_high1', 'AdamSWT_N3_high2',
                    'AdamSWT_N3_high3', 'AdamSWT_N3_high4', 'AdamSWT_N3_high5',
                   'AdamSWT_N3_high6',
                    'AdamSWT_N3_high7', 'AdamSWT_N3_high8',
                    'AdamSWT_N3_high9',
                    'AdamSWT_N3_high10'
                    ]
# multi_head_list = [True, True, False, False, True, True, False, False]
# legends = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6']
legends = ['1, 0', '1, 0.01', '2, 0', '2, 0.01', '3, 0', '3, 0.01',
           '4, 0', '4, 0.01', '5, 0', '5, 0.01', '6, 0', '6, 0.01',
           '7, 0', '7, 0.01', '8, 0', '8, 0.01', '9, 0', '9, 0.01', '10, 0', '10, 0.01']
#############################################
methods = [Joint(), SI(), EWC(), LWF(), EBLL()]
# for method in methods:
for gridsearch_name in gridsearch_names:

    method = IMM('mode')
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
                        img_extention=img_extention, taskcount=5, n_iters=1, gridsearch_name=gridsearch_name)
