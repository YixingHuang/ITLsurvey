import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


cmd1 = 'py ./framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name Adam_FT_unbalancedDataAmount --method_name SI --ds_name tiny   --num_epochs 50   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount --optimizer 1  --fixed_init_lr 0.001 --unbalanced_data'  # --ini_path first_task_basemodel_ICL2 --stochastic

# Adam  FT
cmd20 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --optimizer 1  --fixed_init_lr 0.001   '
cmd21 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --optimizer 1  --fixed_init_lr 0.001   '

cmd22 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat2 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '
cmd23 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat2 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '

cmd24 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat3 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '
cmd25 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat3 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '

cmd26 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat4 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '
cmd27 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat4 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '

cmd28 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat5 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '
cmd29 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '

#Adam SI
cmd30 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --optimizer 1  --fixed_init_lr 0.001   '
cmd31 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --optimizer 1  --fixed_init_lr 0.001   '

cmd32 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat2 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '
cmd33 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat2 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001   '

cmd34 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat3 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '
cmd35 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat3 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001   '

cmd36 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat4 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '
cmd37 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat4 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001   '

cmd38 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat5 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '
cmd39 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_unbalancedDataAmount_repeat5 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_FT_unbalancedDataAmount    --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001   '

## SGD SI
cmd2 = 'py ./framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name SGD_unbalancedData --method_name SI --ds_name tiny   --num_epochs 50   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData --optimizer 0  --fixed_init_lr 0.1'


cmd40 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '
cmd41 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '

cmd42 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat2 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 1 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '
cmd43 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat2 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 1 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '

cmd44 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat3 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 2 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '
cmd45 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat3 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 2 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '

cmd46 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat4 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 3 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '
cmd47 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat4 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 3 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '

cmd48 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat5 --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 4 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '
cmd49 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat5 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 4 --optimizer 0  --fixed_init_lr 0.1  --hyperparams 1 '


cmd50 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --optimizer 0  --fixed_init_lr 0.1   '
cmd51 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --optimizer 0  --fixed_init_lr 0.1   '

cmd52 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat2 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 1 --optimizer 0  --fixed_init_lr 0.1   '
cmd53 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat2 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 1 --optimizer 0  --fixed_init_lr 0.1   '

cmd54 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat3 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 2 --optimizer 0  --fixed_init_lr 0.1   '
cmd55 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat3 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 2 --optimizer 0  --fixed_init_lr 0.1   '

cmd56 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat4 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 3 --optimizer 0  --fixed_init_lr 0.1   '
cmd57 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat4 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 3 --optimizer 0  --fixed_init_lr 0.1   '

cmd58 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat5 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 4 --optimizer 0  --fixed_init_lr 0.1   '
cmd59 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name SGD_unbalancedData_repeat5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_SGD_unbalancedData    --stochastic --seed 4 --optimizer 0  --fixed_init_lr 0.1   '

# cmds = [cmd1]
# cmds = [cmd20, cmd21, cmd18, cmd19]

cmds = [cmd1, cmd20, cmd21, cmd30, cmd31,  cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29, cmd32, cmd33, cmd34, cmd35, cmd36, cmd37, cmd38, cmd39]
# cmds = [cmd30, cmd31, cmd32, cmd33, cmd34, cmd35, cmd36, cmd37, cmd38, cmd39, cmd2, cmd40, cmd41, cmd42, cmd43, cmd44, cmd45, cmd46, cmd47, cmd48, cmd49, cmd50, cmd51, cmd52, cmd53, cmd54, cmd55, cmd56, cmd57, cmd58, cmd59]
# cmds = [cmd40, cmd41, cmd42, cmd43, cmd44, cmd45, cmd46, cmd47, cmd48, cmd49]
for cmd in cmds:
    os.system(cmd)

