import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cmd0 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5 --method_name  joint  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --noisy_center 5'

cmd1 = 'py  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name AdamSWT_N5 --method_name SI --ds_name tiny   --num_epochs 50  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001  --num_class 10  --no_maximal_plasticity_search'

cmd2 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd3 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd4 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.05 --num_class 10  --no_maximal_plasticity_search --hyperparams 400'

cmd5 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5 --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd6 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1'

cmd7 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5 --method_name  meanIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder  first_task_AdamSWT_N5  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search' #Methods Line 819 No_fremework True

cmd8 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5 --method_name  modeIMM --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7  --first_task_basemodel_folder  first_task_AdamSWT_N5  --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search'

cmd9 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5 --method_name  EBLL  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --no_maximal_plasticity_search'

cmd10 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5 --method_name  IT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --drop_margin 0.5 --no_maximal_plasticity_search'

cmd11 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat2 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 100 --noisy_center 5'

cmd13 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat2 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 100 --noisy_center 5'

cmd14 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat2 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 100 --noisy_center 5'

cmd16 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat2 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 100 --noisy_center 5'


cmd21 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat6 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 200 --noisy_center 5'

cmd23 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat6 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 200 --noisy_center 5'

cmd24 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat6 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 200 --noisy_center 5'

cmd26 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat6 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 200 --noisy_center 5'


cmd31 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat3 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 300 --noisy_center 5'

cmd33 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat3 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 300 --noisy_center 5'

cmd34 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat3 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 300 --noisy_center 5'

cmd36 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat3 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 300 --noisy_center 5'


cmd41 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat4 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 400 --noisy_center 5'

cmd43 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat4 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 400 --noisy_center 5'

cmd44 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat4 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 400 --noisy_center 5'

cmd46 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat4 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 400 --noisy_center 5'


cmd51 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 500 --noisy_center 5'

cmd53 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat5 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 500 --noisy_center 5'

cmd54 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat5 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 500 --noisy_center 5'

cmd56 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat5 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 500 --noisy_center 5'


cmd61 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat7 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 1000 --noisy_center 5'

cmd63 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat7 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 1000 --noisy_center 5'

cmd64 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat7 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 1000 --noisy_center 5'

cmd66 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat7 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 1000 --noisy_center 5'


cmd71 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat8 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 2000 --noisy_center 5'

cmd73 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat8 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 2000 --noisy_center 5'

cmd74 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat8 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 2000 --noisy_center 5'

cmd76 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat8 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 2000 --noisy_center 5'


cmd81 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat9 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 3000 --noisy_center 5'

cmd83 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat9 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 3000 --noisy_center 5'

cmd84 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat9 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 3000 --noisy_center 5'

cmd86 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat9 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 3000 --noisy_center 5'


cmd91 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat10 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 4000 --noisy_center 5'

cmd93 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat10 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 4000 --noisy_center 5'

cmd94 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat10 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 4000 --noisy_center 5'

cmd96 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat10 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 4000 --noisy_center 5'


cmd101 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat11 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --stochastic --seed 5000 --noisy_center 5'

cmd103 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat11 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 5000 --noisy_center 5'

cmd104 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat11 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10  --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 5000 --noisy_center 5'

cmd106 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name AdamSWT_N5_repeat11 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder  first_task_AdamSWT_N5 --optimizer 1  --fixed_init_lr 0.001 --num_class 10   --no_maximal_plasticity_search --hyperparams 1 --stochastic --seed 5000 --noisy_center 5'

# cmds = [cmd0, cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10]
# cmds = [cmd0, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9]
#redo SI MAS,
# cmds = [cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10]
# cmds = [cmd13, cmd16]
#cmds = [cmd11, cmd21, cmd23, cmd24, cmd26, cmd31, cmd33, cmd34, cmd36, cmd41, cmd43, cmd44, cmd46, cmd51, cmd53, cmd54, cmd56]
cmds = [cmd61, cmd63, cmd64, cmd66, cmd71, cmd73, cmd74, cmd76, cmd81, cmd83, cmd84, cmd86, cmd91, cmd93, cmd94, cmd96, cmd101, cmd103, cmd104, cmd106]
for cmd in cmds:
 os.system(cmd)

