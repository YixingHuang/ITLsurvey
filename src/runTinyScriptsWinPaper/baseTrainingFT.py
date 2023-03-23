import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


cmd1 = 'py  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name base_training --method_name SI --ds_name tiny   --num_epochs 50  --first_task_basemodel_folder first_task_base_training --optimizer 0  --fixed_init_lr 0.001  --num_class 10 --multi_head --renew_optimizer --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4'

cmd2 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10 --multi_head --renew_optimizer'

cmd3 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_reloadOp --method_name  FT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_reloadOp    --optimizer 0  --fixed_init_lr 0.05 --num_class 10 --multi_head --no_maximal_plasticity_search'

cmd4 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  FT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_reloadOp    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd22 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_renewOp_SH --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10 --renew_optimizer --no_maximal_plasticity_search'

cmd5 = 'py  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name base_training --method_name SI --ds_name tiny   --num_epochs 50  --first_task_basemodel_folder first_task_base_training_paper --optimizer 0  --fixed_init_lr 0.1  --num_class 10 --multi_head --renew_optimizer --boot_lr_grid 1e-1 --no_maximal_plasticity_search --hyperparams 0 --stochastic --seed 2'

cmd6 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_paper --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_paper    --optimizer 0  --fixed_init_lr 0.05 --num_class 10 --multi_head --renew_optimizer --no_maximal_plasticity_search --stochastic --seed 2'

cmd7 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_reloadOp_paper --method_name  FT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_paper    --optimizer 0  --fixed_init_lr 0.05 --num_class 10 --multi_head --no_maximal_plasticity_search --stochastic --seed 2'

cmd8 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH_paper --method_name  FT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_paper    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 2'


cmd10 = 'py  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name base_training_Adam_paper --method_name SI --ds_name tiny   --num_epochs 50  --first_task_basemodel_folder first_task_base_training_Adam_paper --optimizer 1  --fixed_init_lr 0.001  --num_class 10 --multi_head --renew_optimizer --boot_lr_grid 1e-3 --no_maximal_plasticity_search --hyperparams 0'

cmd12 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_Adam_paper --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_Adam_paper   --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --no_maximal_plasticity_search'

cmd13 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_reloadOp_Adam_paper_redo --method_name  FT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_Adam_paper    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --multi_head --no_maximal_plasticity_search --stochastic --seed 2'

cmd14 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH_Adam_paper --method_name  FT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_Adam_paper    --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd15 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_renewOp_SH_Adam_paper --method_name  FT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_Adam_paper    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --renew_optimizer --no_maximal_plasticity_search'

# cmds = [cmd0, cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9]
# cmds = [cmd10, cmd12, cmd13, cmd14]
# cmds = [cmd5, cmd6, cmd7, cmd8]
cmds = [cmd22, cmd15]
for cmd in cmds:
    os.system(cmd)

