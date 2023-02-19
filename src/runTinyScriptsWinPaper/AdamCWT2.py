import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


cmd31 = 'python  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump  --gridsearch_name adam_CWT_ep5 --method_name SI --ds_name tiny   --num_epochs 5 --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001  --num_class 10  --no_maximal_plasticity_search'

cmd32 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd33 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd34 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd35 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd36 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search'

cmd37 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  meanIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10 --seed 7  --first_task_basemodel_folder  first_task_adam_CWT_ep5  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd38 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  modeIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10 --seed 7  --first_task_basemodel_folder  first_task_adam_CWT_ep5  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd39 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT_ep5 --method_name  EBLL  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 5 --n_iters 10  --first_task_basemodel_folder  first_task_adam_CWT_ep5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --no_maximal_plasticity_search'

cmds = [cmd31, cmd32, cmd33, cmd34, cmd35, cmd36, cmd37, cmd38, cmd39]

for cmd in cmds:
 os.system(cmd)

