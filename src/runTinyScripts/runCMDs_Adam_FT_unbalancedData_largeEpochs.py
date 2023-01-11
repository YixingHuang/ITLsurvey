import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


cmd1 = 'python ./framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name Adam_FT_unbalancedData_long --method_name SI --ds_name tiny   --num_epochs 150   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long --optimizer 1  --fixed_init_lr 0.001'  # --ini_path first_task_basemodel_ICL2 --stochastic

# Adam  FT
cmd18 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long --method_name  FT  --ds_name tiny   --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --optimizer 1  --fixed_init_lr 0.001   '
cmd19 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --optimizer 1  --fixed_init_lr 0.001   '

cmd22 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat2 --method_name  FT  --ds_name tiny   --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '
cmd23 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat2 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '

cmd24 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat3 --method_name  FT  --ds_name tiny   --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '
cmd25 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat3 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '

cmd26 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat4 --method_name  FT  --ds_name tiny   --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '
cmd27 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat4 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '

cmd28 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat5 --method_name  FT  --ds_name tiny   --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '
cmd29 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '

#Adam SI
cmd30 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long --method_name  SI  --ds_name tiny   --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --optimizer 1  --fixed_init_lr 0.001   '
cmd31 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --optimizer 1  --fixed_init_lr 0.001   '

cmd32 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat2 --method_name  SI  --ds_name tiny   --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '
cmd33 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat2 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '

cmd34 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat3 --method_name  SI  --ds_name tiny   --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '
cmd35 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat3 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '

cmd36 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat4 --method_name  SI  --ds_name tiny   --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '
cmd37 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat4 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '

cmd38 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat5 --method_name  SI  --ds_name tiny   --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '
cmd39 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_long_repeat5 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 150 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData_long   --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '


cmds = [cmd1, cmd18, cmd19, cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29, cmd30, cmd31, cmd32, cmd33, cmd34, cmd35, cmd36, cmd37, cmd38, cmd39]


for cmd in cmds:
    os.system(cmd)

