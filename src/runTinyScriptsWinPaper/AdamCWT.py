import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


cmd21 = 'python  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump  --gridsearch_name adam_CWT --method_name SI --ds_name tiny   --num_epochs 10 --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001  --num_class 10  --no_maximal_plasticity_search'

cmd22 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd23 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd24 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search'

cmd25 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd26 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search'

cmd27 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  meanIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7  --first_task_basemodel_folder  first_task_adam_CWT  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd28 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  modeIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7  --first_task_basemodel_folder  first_task_adam_CWT  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd29 = 'python ../framework/main.py small_VGG9_cl_128_128   --gridsearch_name adam_CWT --method_name  EBLL  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --first_task_basemodel_folder  first_task_adam_CWT --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --no_maximal_plasticity_search'

cmds = [cmd21, cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29]

for cmd in cmds:
 os.system(cmd)

