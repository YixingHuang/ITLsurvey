import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cmd0 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  joint  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.001 --num_class 10 --renew_optimizer --lr_grid 0.1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4'

cmd1 = 'py  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name base_training --method_name SI --ds_name tiny   --num_epochs 50  --first_task_basemodel_folder first_task_base_training --optimizer 0  --fixed_init_lr 0.001  --num_class 10 --multi_head --renew_optimizer --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4'

cmd2 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer'

cmd3 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --lr_grid 0.1,5e-2,1e-2,5e-3,1e-3'

cmd4 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.05 --num_class 10 --multi_head --renew_optimizer --lr_grid 0.1,5e-2,1e-2,5e-3,1e-3'

cmd5 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer'

cmd6 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer'

cmd7 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  meanIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder first_task_base_training  --optimizer 0  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer' #Methods Line 819 No_fremework True

cmd8 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  modeIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder first_task_base_training  --optimizer 0  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer'

cmd9 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  EBLL  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --static_hyperparams 0.01,0.001;50;1e-1,1e-2;100,300'

cmd10 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training --method_name  IT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training    --optimizer 0  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --drop_margin 0.5 --no_maximal_plasticity_search'

# cmds = [cmd0, cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9]
cmds = [cmd4]
for cmd in cmds:
    os.system(cmd)

