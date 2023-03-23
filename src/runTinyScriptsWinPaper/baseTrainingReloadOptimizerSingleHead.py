import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# cmd0 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH --method_name  joint  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_reloadOp_SH    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --lr_grid 0.1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4'

cmd1 = 'py  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name  base_training_reloadOp_SH --method_name SI --ds_name tiny   --num_epochs 50  --first_task_basemodel_folder first_task_base_training_reloadOp_SH --optimizer 0  --fixed_init_lr 0.05  --num_class 10   --boot_lr_grid 1e-1,5e-2'

cmd2 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH --method_name  FT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_reloadOp_SH    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd3 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH --method_name  SI  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_reloadOp_SH    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams_seq 0,0.1,0.1,0.1,0.1'

cmd4 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH --method_name  EWC  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_reloadOp_SH    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 1'

cmd5 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH --method_name  LWF  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_reloadOp_SH    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams_seq 0,1,1,1,1'

cmd6 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH --method_name  MAS --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_reloadOp_SH    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams_seq 0,0.1,0.1,0.1,0.1'

cmd7 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH --method_name  meanIMM --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder first_task_base_training_reloadOp_SH  --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd8 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH --method_name  modeIMM --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder first_task_base_training_reloadOp_SH  --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd9 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH --method_name  EBLL  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_reloadOp_SH    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 1;0.1'


# cmd10 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SH --method_name  IT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_reloadOp_SH    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --drop_margin 0.5'

# cmds = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9]
cmds = [cmd4]
for cmd in cmds:
    os.system(cmd)

