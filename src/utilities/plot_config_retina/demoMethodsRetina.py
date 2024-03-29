# RUN THIS AS INIT
from data.dataset import *

from models.net import *
from utilities.main_postprocessing import *

# CONFIG
config = utils.get_parsed_config()
test_results_root_path = utils.read_from_config(config, 'test_results_root_path')
tr_results_root_path = utils.read_from_config(config, 'tr_results_root_path')
models_root_path = utils.read_from_config(config, 'models_root_path')

dataset = RetinalFundusDataset()
model = SmallVGG9(models_root_path, dataset.input_size)
# model = BaseVGG9(models_root_path, dataset.input_size)
# model = DeepVGG22(models_root_path, dataset.input_size)
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
gridsearch_name = "retina_try"
#############################################
# methods = [FineTuning(), SI(), EWC(), MAS()]
methods = [Joint(), FineTuning(), EWC(), SI(), LWF(), EBLL(), IMM('mean'), IMM('mode')]

for method in methods:
    # method = SI()
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
                    plot_seq_forgetting=plot_seq_forgetting, save_img_parent_dir=out_name, img_extention=img_extention,
                        taskcount=5, n_iters=1, gridsearch_name=gridsearch_name, multi_head=None)
