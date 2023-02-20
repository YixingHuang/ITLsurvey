import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# cmd0 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  joint  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --lr_grid 0.1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4'

cmd1 = 'py  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name  base_training_reloadOp_SHRedo --method_name SI --ds_name tiny   --num_epochs 50 --first_task_basemodel_folder first_task_base_training --optimizer 0  --fixed_init_lr 0.05  --num_class 10   --boot_lr_grid 1e-1,5e-2'

cmd2 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  FT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd3 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  SI  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams_seq 0,0.1,0.1,0.1,0.1'

cmd4 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  EWC  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 1'

cmd5 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  LWF  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams_seq 0,1,1,1,1'

cmd6 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  MAS --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams_seq 0,0.1,0.1,0.1,0.1'

cmd7 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  meanIMM --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --seed 7 --first_task_basemodel_folder first_task_base_training  --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd8 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  modeIMM --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --seed 7 --first_task_basemodel_folder first_task_base_training  --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd9 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  EBLL  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 1;0.1'


# cmd10 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  base_training_reloadOp_SHRedo --method_name  IT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 50 --n_iters 1 --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --drop_margin 0.5'

cmd11 = 'py  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name  SGD_CWT --method_name SI --ds_name tiny   --num_epochs 10  --first_task_basemodel_folder first_task_SGD_CWT --optimizer 0  --fixed_init_lr 0.05  --num_class 10   --boot_lr_grid 1e-1,5e-2'

cmd12 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  SGD_CWT --method_name  FT  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 10 --n_iters 5  --first_task_basemodel_folder first_task_SGD_CWT    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd13 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  SGD_CWT --method_name  SI  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 10 --n_iters 5  --first_task_basemodel_folder first_task_SGD_CWT    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 0.1'

cmd14 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  SGD_CWT --method_name  EWC  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 10 --n_iters 5  --first_task_basemodel_folder first_task_SGD_CWT    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 1'

cmd15 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  SGD_CWT --method_name  LWF  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 10 --n_iters 5  --first_task_basemodel_folder first_task_SGD_CWT    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 1'

cmd16 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  SGD_CWT --method_name  MAS --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 10 --n_iters 5  --first_task_basemodel_folder first_task_SGD_CWT    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 0.1'

cmd17 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  SGD_CWT --method_name  meanIMM --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 10 --n_iters 5 --seed 7  --first_task_basemodel_folder first_task_SGD_CWT  --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd18 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  SGD_CWT --method_name  modeIMM --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 10 --n_iters 5 --seed 7  --first_task_basemodel_folder first_task_SGD_CWT  --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd19 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name  SGD_CWT --method_name  EBLL  --ds_name tiny    --test  --test_overwrite_mode    --num_epochs 10 --n_iters 5  --first_task_basemodel_folder first_task_SGD_CWT    --optimizer 0  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 1;0.1'


# cmds = [cmd11, cmd12, cmd13, cmd14, cmd15, cmd16, cmd17, cmd18, cmd19]
cmds = [cmd11, cmd12, cmd13, cmd14, cmd15, cmd16, cmd17, cmd18, cmd19]

for cmd in cmds:
    os.system(cmd)

