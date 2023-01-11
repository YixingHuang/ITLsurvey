import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
cmd1 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name STCL_repeat1 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50'
cmd2 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name STCL_repeat1 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --stochastic'
cmd3 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name STCL_repeat1 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --stochastic'

cmd4 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name STCL_repeat2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 1 --stochastic'
cmd5 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name STCL_repeat2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 1 --stochastic'

cmd6 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name STCL_repeat3 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 2 --stochastic'
cmd7 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name STCL_repeat3 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 2 --stochastic'

cmd8 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name STCL_repeat4 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 3 --stochastic'
cmd9 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name STCL_repeat4 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 3 --stochastic'

cmd10 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name STCL_repeat5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 4 --stochastic'
cmd11 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name STCL_repeat5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 4 --stochastic'
#

cmd50 = 'python  ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name STCL_repeat1 --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 7 --stochastic'

cmd51 = 'python  ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name STCL_repeat1 --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --test_max_task_count 5 --num_epochs 50 --n_iters 1 --seed 7 --stochastic'

cmd52 = 'python  ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name STCL_repeat2 --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 1 --stochastic'

cmd53 = 'python  ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name STCL_repeat2 --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --test_max_task_count 5 --num_epochs 50 --n_iters 1 --seed 1 --stochastic'

cmd54 = 'python  ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name STCL_repeat3 --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 2 --stochastic'

cmd55 = 'python  ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name STCL_repeat3 --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --test_max_task_count 5 --num_epochs 50 --n_iters 1 --seed 2 --stochastic'

cmd56 = 'python  ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name STCL_repeat4 --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 3 --stochastic'

cmd57 = 'python  ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name STCL_repeat4 --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --test_max_task_count 5 --num_epochs 50 --n_iters 1 --seed 3 --stochastic'

cmd58 = 'python  ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name STCL_repeat5 --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1 --seed 4 --stochastic'

cmd59 = 'python  ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name STCL_repeat5 --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --test_max_task_count 5 --num_epochs 50 --n_iters 1 --seed 4 --stochastic'


# cmd22 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1'
#
# cmd23 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 50 --n_iters 1'
#
# cmd24 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1'
#
# cmd25 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 50 --n_iters 1'
#
#
# cmd4 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --num_epochs 50 --n_iters 1'
#
# cmd5 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 50 --n_iters 1'
#
# # cmd6 = 'python  ./utilities/plot_configs/demoMAS.py'
#
# cmd7 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1'
#
# cmd8 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 50 --n_iters 1'
#
# # cmd9 = 'python  ./utilities/plot_configs/demoLWF.py'
#
# cmd10 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --n_iters 1'
#
# cmd11 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 50 --n_iters 1'

# cmds = [cmd1]
# cmds = [cmd2, cmd3]
# cmds = [cmd22, cmd23, cmd24, cmd25]
# cmds = [cmd4, cmd5, cmd7, cmd8, cmd10, cmd11]
# cmds = [cmd4, cmd5, cmd10, cmd11]
# cmds = [cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10, cmd11]
# cmds = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10, cmd11, cmd50, cmd51, cmd52, cmd53, cmd54, cmd55, cmd56, cmd57, cmd58, cmd59]
# cmds = [cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10, cmd11]
cmds = [cmd52, cmd53, cmd54, cmd55, cmd56, cmd57, cmd58, cmd59]
for cmd in cmds:
    os.system(cmd)

