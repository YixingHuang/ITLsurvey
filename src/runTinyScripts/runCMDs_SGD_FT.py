import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


cmd1 = 'python ./framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name SGD_FT_correct --method_name SI --ds_name tiny   --num_epochs 50 --no_maximal_plasticity_search --optimizer 0  --fixed_init_lr 0.1'  # --ini_path first_task_basemodel_ICL2 --stochastic

cmd18 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_FT_correct --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --optimizer 0  --fixed_init_lr 0.1'
cmd19 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_FT_correct --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --optimizer 0  --fixed_init_lr 0.1'

cmd22 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_FT_correct_repeat2 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 1 --optimizer 0  --fixed_init_lr 0.1'
cmd23 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_FT_correct_repeat2 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 1 --optimizer 0  --fixed_init_lr 0.1'

cmd24 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_FT_correct_repeat3 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 2 --optimizer 0  --fixed_init_lr 0.1'
cmd25 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_FT_correct_repeat3 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 2 --optimizer 0  --fixed_init_lr 0.1'

cmd26 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_FT_correct_repeat4 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 3 --optimizer 0  --fixed_init_lr 0.1'
cmd27 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_FT_correct_repeat4 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 3 --optimizer 0  --fixed_init_lr 0.1'

cmd28 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_FT_correct_repeat5 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 4 --optimizer 0  --fixed_init_lr 0.1'
cmd29 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_FT_correct_repeat5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --stochastic --seed 4 --optimizer 0  --fixed_init_lr 0.1'


# cmds = [cmd1]
# cmds = [cmd20, cmd21, cmd18, cmd19]
cmds = [cmd1, cmd18, cmd19, cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29]
for cmd in cmds:
    os.system(cmd)

