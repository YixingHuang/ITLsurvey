import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# cmd0 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH --method_name  joint  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --renew_optimizer --no_maximal_plasticity_search'

#The basic model uses the same model for base_training_adam
# cmd1 = 'python  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name base_training_adam_SH --method_name SI --ds_name tiny   --num_epochs 50  --first_task_basemodel_folder  first_task_base_training_adam --optimizer 1  --fixed_init_lr 0.001  --num_class 10   --renew_optimizer --no_maximal_plasticity_search'

cmd2 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --renew_optimizer --no_maximal_plasticity_search'

cmd3 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --renew_optimizer --no_maximal_plasticity_search'

cmd4 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.05 --num_class 10   --renew_optimizer --no_maximal_plasticity_search --hyperparams 0'

cmd5 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --renew_optimizer --no_maximal_plasticity_search'

cmd6 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10    --renew_optimizer --no_maximal_plasticity_search'

cmd7 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH --method_name  meanIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder  first_task_base_training_adam  --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --renew_optimizer --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd8 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH --method_name  modeIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder  first_task_base_training_adam  --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --renew_optimizer --no_maximal_plasticity_search'

cmd9 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH --method_name  EBLL  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --renew_optimizer --no_maximal_plasticity_search --hyperparams 1;0.1'
cmd11 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH_repeat --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --renew_optimizer --no_maximal_plasticity_search --stochastic --seed 1'

# cmd10 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam_SH --method_name  IT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --renew_optimizer --drop_margin 0.5 --no_maximal_plasticity_search'

# cmds = [cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9]


cmd21 = 'python  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump  --gridsearch_name adam_CWT --method_name SI --ds_name tiny   --num_epochs 10 --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001  --num_class 10  --no_maximal_plasticity_search'

cmd22 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd23 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd24 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd25 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd26 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search'

cmd27 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  meanIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7  --first_task_basemodel_folder  first_task_adam_CWT  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd28 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  modeIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7  --first_task_basemodel_folder  first_task_adam_CWT  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd29 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  EBLL  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --no_maximal_plasticity_search'

# cmds = [cmd21, cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29]


cmd31 = 'python  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump  --gridsearch_name adam_CWT_ep5 --method_name SI --ds_name tiny   --num_epochs 5 --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001  --num_class 10  --no_maximal_plasticity_search'

cmd32 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd33 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd34 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd35 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd36 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search'

cmd37 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  meanIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10 --seed 7  --first_task_basemodel_folder  first_task_adam_CWT_ep5  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd38 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  modeIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10 --seed 7  --first_task_basemodel_folder  first_task_adam_CWT_ep5  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd39 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  EBLL  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --no_maximal_plasticity_search'

# cmds = [cmd31, cmd32, cmd33, cmd34, cmd35, cmd36, cmd37, cmd38, cmd39]


cmds = [cmd9, cmd4, cmd21, cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29, cmd31, cmd32, cmd33, cmd34, cmd35, cmd36, cmd37, cmd38, cmd39]
for cmd in cmds:
    os.system(cmd)

