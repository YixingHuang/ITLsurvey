import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
cmd1 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name MC5ICL5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10'
cmd2 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat1 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 7'
cmd3 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat1 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 7'

cmd4 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 1'
cmd5 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat2 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 1'

cmd6 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat3 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 2'
cmd7 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat3 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 2'

cmd8 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat4 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 3'
cmd9 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat4 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 3'

cmd10 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --seed 4'
cmd11 = 'python  ../framework/main.py  small_VGG9_cl_128_128  --gridsearch_name repeat5 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --seed 4'
#
# cmd22 = 'python  ../framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5'
#
# cmd23 = 'python  ../framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5'
#
# cmd24 = 'python  ../framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5'
#
# cmd25 = 'python  ../framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5'
#
#
# cmd4 = 'python  ../framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --num_epochs 10 --n_iters 5'
#
# cmd5 = 'python  ../framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5'
#
# # cmd6 = 'python  ./utilities/plot_configs/demoMAS.py'
#
# cmd7 = 'python  ../framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5'
#
# cmd8 = 'python  ../framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5'
#
# # cmd9 = 'python  ./utilities/plot_configs/demoLWF.py'
#
# cmd10 = 'python  ../framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5'
#
# cmd11 = 'python  ../framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5 --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5'

# cmds = [cmd1]
# cmds = [cmd2, cmd3]
# cmds = [cmd22, cmd23, cmd24, cmd25]
# cmds = [cmd4, cmd5, cmd7, cmd8, cmd10, cmd11]
# cmds = [cmd4, cmd5, cmd10, cmd11]
# cmds = [cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10, cmd11]
cmds = [cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10, cmd11]
for cmd in cmds:
    os.system(cmd)

