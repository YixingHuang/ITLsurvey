import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
cmd1 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name MC5ICL5_noReplace --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10'
cmd2 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name MC5ICL5_noReplace --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --num_epochs_initial_lr 50'
cmd3 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name MC5ICL5_noReplace --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5  --num_epochs_initial_lr 50'

cmd22 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5_noMPS --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --num_epochs 10 --n_iters 5 --no_maximal_plasticity_search --num_epochs_initial_lr 50'

cmd23 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5_noMPS --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --no_maximal_plasticity_search --num_epochs_initial_lr 50'

cmd24 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5_noMPS --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --num_epochs 10 --n_iters 5 --no_maximal_plasticity_search --num_epochs_initial_lr 50'

cmd25 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5_noMPS --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --no_maximal_plasticity_search --num_epochs_initial_lr 50'


cmd4 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5_noMPS --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4   --num_epochs 10 --n_iters 5 --no_maximal_plasticity_search --num_epochs_initial_lr 50'

cmd5 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5_noMPS --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --no_maximal_plasticity_search --num_epochs_initial_lr 50'

# cmd6 = 'python  ./utilities/plot_configs/demoMAS.py'

cmd7 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5_noMPS --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --num_epochs 10 --n_iters 5 --no_maximal_plasticity_search --num_epochs_initial_lr 50'

cmd8 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5_noMPS --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --no_maximal_plasticity_search --num_epochs_initial_lr 50'

# cmd9 = 'python  ./utilities/plot_configs/demoLWF.py'

cmd10 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5_noMPS --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --num_epochs 10 --n_iters 5 --no_maximal_plasticity_search --num_epochs_initial_lr 50'

cmd11 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5_noMPS --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --no_maximal_plasticity_search --num_epochs_initial_lr 50'

# cmds = [cmd1]
# cmds = [cmd2, cmd3]
# cmds = [cmd22, cmd23, cmd24, cmd25]
# cmds = [cmd4, cmd5, cmd7, cmd8, cmd10, cmd11]
# cmds = [cmd4, cmd5, cmd10, cmd11]
# cmds = [cmd1, cmd2, cmd3, cmd22, cmd23, cmd24, cmd25, cmd4, cmd5]
# cmds = [cmd1, cmd2, cmd3]
cmds = [cmd3]
for cmd in cmds:
    os.system(cmd)

