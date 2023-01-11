import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


cmd1 = 'python ./framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name Adam_FT_renewOpt2 --method_name SI --ds_name tiny   --num_epochs 50 --no_maximal_plasticity_search --optimizer 1  --fixed_init_lr 0.001'  # --ini_path first_task_basemodel_ICL2 --stochastic

cmd18 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_renewOpt2 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam_STCL_repeat --optimizer 1  --fixed_init_lr 0.001  --renew_optimizer'
cmd19 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_renewOpt2 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam_STCL_repeat --optimizer 1  --fixed_init_lr 0.001  --renew_optimizer'

cmd22 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_renewOpt2_repeat2 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam_STCL_repeat --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001  --renew_optimizer'
cmd23 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_renewOpt2_repeat2 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam_STCL_repeat --stochastic --seed 1 --optimizer 1  --fixed_init_lr 0.001  --renew_optimizer'

cmd24 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_renewOpt2_repeat3 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam_STCL_repeat --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001  --renew_optimizer'
cmd25 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_renewOpt2_repeat3 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam_STCL_repeat --stochastic --seed 2 --optimizer 1  --fixed_init_lr 0.001  --renew_optimizer'

cmd26 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_renewOpt2_repeat4 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam_STCL_repeat --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001  --renew_optimizer'
cmd27 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_renewOpt2_repeat4 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam_STCL_repeat --stochastic --seed 3 --optimizer 1  --fixed_init_lr 0.001  --renew_optimizer'

cmd28 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_renewOpt2_repeat5 --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam_STCL_repeat --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001  --renew_optimizer'
cmd29 = 'python ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_FT_renewOpt2_repeat5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam_STCL_repeat --stochastic --seed 4 --optimizer 1  --fixed_init_lr 0.001  --renew_optimizer'


# cmds = [cmd1]
# cmds = [cmd20, cmd21, cmd18, cmd19]
cmds = [cmd1, cmd18, cmd19, cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29]

for cmd in cmds:
    os.system(cmd)

