import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
cmd1 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name STCL_Adam --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50  --no_maximal_plasticity_search --ini_path first_task_basemodel_ICL2'
cmd2 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_RmseProp_NoMPS --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'
cmd3 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_RmseProp_NoMPS --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7'


cmd4 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_RmseProp --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7'
cmd5 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_RmseProp --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7'


cmd6 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'
cmd7 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'


cmd8 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS_initFreeze --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'
cmd9 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS_initFreeze --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'

cmd10 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS_initFreeze_smallerLR --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'
cmd11 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS_initFreeze_smallerLR --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'


cmd12 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS_initFreeze_smallerLR2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'
cmd13 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS_initFreeze_smallerLR2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'

cmd14 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS_initFreeze_noPop --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'
cmd15 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS_initFreeze_noPop --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'

cmd16 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS_initFreeze_bigLambda --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'
cmd17 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_NoMPS_initFreeze_bigLambda --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search'

cmd18 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam'
cmd19 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam'

cmd20 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam'
cmd21 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam'

cmd22 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 1'
cmd23 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 1'

cmd24 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat3 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 2'
cmd25 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat3 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 2'

cmd26 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat4 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 3'
cmd27 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat4 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 3'

cmd28 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 4'
cmd29 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 4'

cmd30 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat2 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 1'
cmd31 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat2 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 1'

cmd32 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat3 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 2'
cmd33 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat3 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 2'

cmd34 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat4 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 3'
cmd35 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat4 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 3'

cmd36 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat5 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 4'
cmd37 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam_repeat5 --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 4'



# cmd38 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 1'
# cmd39 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name optimizer_Adam --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel_adam --stochastic --seed 1'



# cmd4 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 1'
# cmd5 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 1'
#
# cmd6 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat3 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 2'
# cmd7 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat3 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 2'
#
# cmd8 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat4 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 3'
# cmd9 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat4 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 3'
#
# cmd10 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 4'
# cmd11 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 4'
# #
# cmd22 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5'
#
# cmd23 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5'
#
# cmd24 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5'
#
# cmd25 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5'
#
#
# cmd4 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --num_epochs 10 --n_iters 5'
#
# cmd5 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5'
#
# # cmd6 = 'python  ./utilities/plot_configs/demoMAS.py'
#
# cmd7 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5'
#
# cmd8 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5'
#
# # cmd9 = 'python  ./utilities/plot_configs/demoLWF.py'
#
# cmd10 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5'
#
# cmd11 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5'

# cmds = [cmd1]
# cmds = [cmd2, cmd3]
# cmds = [cmd22, cmd23, cmd24, cmd25]
# cmds = [cmd4, cmd5, cmd7, cmd8, cmd10, cmd11]
# cmds = [cmd4, cmd5, cmd10, cmd11]
# cmds = [cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10, cmd11]
# cmds = [cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10, cmd11]
# cmds = [cmd2, cmd3, cmd4, cmd5]
# cmds = [cmd1]
# cmds = [cmd20, cmd21, cmd18, cmd19]
cmds = [cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29, cmd30, cmd31, cmd32, cmd33, cmd34, cmd35, cmd36, cmd37]
for cmd in cmds:
    os.system(cmd)

