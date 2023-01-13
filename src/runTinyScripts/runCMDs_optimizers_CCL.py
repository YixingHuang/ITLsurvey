import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
cmd1 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name optimizer_Adam_CCL_lambda10 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 11 --no_maximal_plasticity_search'  # --ini_path first_task_basemodel_ICL2 --stochastic

cmd18 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel'
cmd19 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel'

cmd20 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel'
cmd21 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel'

cmd22 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 1'
cmd23 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 1'

cmd24 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat3 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 2'
cmd25 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat3 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 2'

cmd26 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat4 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 3'
cmd27 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat4 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 3'

cmd28 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda1_repeat5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 4'
cmd29 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda1_repeat5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 4'

cmd30 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat2 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 1'
cmd31 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat2 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 1'

cmd32 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat3 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 2'
cmd33 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat3 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 2'

cmd34 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat4 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 3'
cmd35 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda10_repeat4 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 3'

cmd36 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda1_repeat5 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 4'
cmd37 = 'python   ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_CCL_lambda1_repeat5 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 4'


# cmds = [cmd1]
# cmds = [cmd20, cmd21, cmd18, cmd19]
cmds = [cmd18, cmd19, cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29]

for cmd in cmds:
    os.system(cmd)

