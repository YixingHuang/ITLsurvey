import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


cmd1 = 'python  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name Adam_FT_unbalancedData --method_name SI --ds_name tiny   --num_epochs 50   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData --optimizer 1  --fixed_init_lr 0.001'  # --ini_path first_task_basemodel_ICL2 --stochastic

# Adam  FT
cmd18 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --optimizer 1  --fixed_init_lr 0.001   '
cmd19 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --optimizer 1  --fixed_init_lr 0.001   '

cmd22 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat2 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '
cmd23 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat2 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '

cmd24 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat3 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '
cmd25 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat3 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '

cmd26 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat4 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '
cmd27 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat4 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '

cmd28 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat5 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '
cmd29 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '

#Adam SI
cmd30 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --optimizer 1  --fixed_init_lr 0.001   '
cmd31 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --optimizer 1  --fixed_init_lr 0.001   '

cmd32 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat2 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '
cmd33 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat2 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '

cmd34 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat3 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '
cmd35 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat3 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '

cmd36 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat4 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '
cmd37 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat4 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '

cmd38 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat5 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '
cmd39 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedData_repeat5 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedData    --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '

## SGD SI
cmd2 = 'python  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name SGD_unbalancedData --method_name SI --ds_name tiny   --num_epochs 50   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData --optimizer 0  --fixed_init_lr 0.1'


cmd40 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '
cmd41 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '

cmd42 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat2 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 1 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '
cmd43 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat2 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 1 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '

cmd44 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat3 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 2 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '
cmd45 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat3 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 2 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '

cmd46 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat4 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 3 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '
cmd47 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat4 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 3 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '

cmd48 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat5 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 4 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '
cmd49 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat5 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 4 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '


cmd50 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --optimizer 0  --fixed_init_lr 0.1   '
cmd51 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --optimizer 0  --fixed_init_lr 0.1   '

cmd52 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat2 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 1 --optimizer 0  --fixed_init_lr 0.1   '
cmd53 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat2 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 1 --optimizer 0  --fixed_init_lr 0.1   '

cmd54 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat3 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 2 --optimizer 0  --fixed_init_lr 0.1   '
cmd55 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat3 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 2 --optimizer 0  --fixed_init_lr 0.1   '

cmd56 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat4 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 3 --optimizer 0  --fixed_init_lr 0.1   '
cmd57 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat4 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 3 --optimizer 0  --fixed_init_lr 0.1   '

cmd58 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat5 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 4 --optimizer 0  --fixed_init_lr 0.1   '
cmd59 = 'python  ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 4 --optimizer 0  --fixed_init_lr 0.1   '

# cmds = [cmd1]
# cmds = [cmd20, cmd21, cmd18, cmd19]

# cmds = [cmd1, cmd18, cmd19, cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29]
# cmds = [cmd30, cmd31, cmd32, cmd33, cmd34, cmd35, cmd36, cmd37, cmd38, cmd39, cmd2, cmd40, cmd41, cmd42, cmd43, cmd44, cmd45, cmd46, cmd47, cmd48, cmd49, cmd50, cmd51, cmd52, cmd53, cmd54, cmd55, cmd56, cmd57, cmd58, cmd59]
cmds = [cmd40, cmd41, cmd42, cmd43, cmd44, cmd45, cmd46, cmd47, cmd48, cmd49]
for cmd in cmds:
    os.system(cmd)

