import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

cmd0 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam --method_name  joint  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --renew_optimizer --no_maximal_plasticity_search'

cmd1 = 'python  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name base_training_adam --method_name SI --ds_name tiny   --num_epochs 50  --first_task_basemodel_folder first_task_base_training_adam --optimizer 1  --fixed_init_lr 0.001  --num_class 10 --multi_head --renew_optimizer --no_maximal_plasticity_search'

cmd2 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --no_maximal_plasticity_search'

cmd3 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --no_maximal_plasticity_search'

cmd4 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.05 --num_class 10 --multi_head --renew_optimizer --no_maximal_plasticity_search --hyperparams 1'

cmd5 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --no_maximal_plasticity_search'

cmd6 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --multi_head  --renew_optimizer --no_maximal_plasticity_search'

cmd7 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam --method_name  meanIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder first_task_base_training_adam  --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd8 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam --method_name  modeIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder first_task_base_training_adam  --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --no_maximal_plasticity_search'

cmd9 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam --method_name  EBLL  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --no_maximal_plasticity_search'

cmd10 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name base_training_adam --method_name  IT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_base_training_adam    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --multi_head --renew_optimizer --drop_margin 0.5 --no_maximal_plasticity_search --stochastic --seed 1'

cmds = [cmd4]

for cmd in cmds:
    os.system(cmd)

