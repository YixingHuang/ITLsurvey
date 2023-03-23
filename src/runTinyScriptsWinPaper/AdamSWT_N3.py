import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cmd0 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N3 --method_name  joint  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N3 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --noisy_center 3'

cmd1 = 'py  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name AdamSWT_N3 --method_name SI --ds_name tiny   --num_epochs 50  --first_task_basemodel_folder  first_task_AdamSWT_N3 --optimizer 1  --fixed_init_lr 0.001  --num_class 10  --no_maximal_plasticity_search'

cmd2 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N3 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N3 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 2'

cmd3 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N3 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N3 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd4 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N3 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N3 --optimizer 1  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 400'

cmd5 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N3 --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N3 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd6 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N3 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N3 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1'

cmd7 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N3 --method_name  meanIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder  first_task_AdamSWT_N3  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd8 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N3 --method_name  modeIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder  first_task_AdamSWT_N3  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd9 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N3 --method_name  EBLL  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N3 --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --no_maximal_plasticity_search'

cmd10 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N3 --method_name  IT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N3 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --drop_margin 0.5 --no_maximal_plasticity_search'

# cmds = [cmd0, cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10]
# cmds = [cmd3, cmd6, cmd7, cmd8, cmd9]
cmds = [cmd4]
for cmd in cmds:
 os.system(cmd)

