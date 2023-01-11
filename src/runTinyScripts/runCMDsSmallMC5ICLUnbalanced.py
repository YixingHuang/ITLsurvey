import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
cmd1 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name MC5ICL5unbalanced --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --unbalanced_data'
cmd2= ' python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name MC5ICL5unbalanced --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --unbalanced_data'
cmd3 = ' python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name MC5ICL5unbalanced --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd4 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name  MC5ICL5unbalanced  --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd5 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name  MC5ICL5unbalanced  --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --unbalanced_data'

# cmd6 = 'python  ./utilities/plot_configs/demoMAS.python'

cmd7 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name  MC5ICL5unbalanced  --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd8 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name  MC5ICL5unbalanced  --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --unbalanced_data'

# cmd9 = 'python  ./utilities/plot_configs/demoLWF.python'

cmd10 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name  MC5ICL5unbalanced  --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd11 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name  MC5ICL5unbalanced  --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd14 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5unbalanced --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --unbalanced_data'

cmd15 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5unbalanced --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --unbalanced_data'


cmd16 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5unbalanced--method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd17 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5unbalanced--method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd18 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5unbalanced --method_name EBLL --ds_name tiny   --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --static_hyperparams 0.01,0.001;50;1e-1,1e-2;100,300 --num_epochs 10 --unbalanced_data'

cmd19 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name MC5ICL5unbalanced --method_name EBLL --ds_name tiny   --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --static_hyperparams 0.01,0.001;50;1e-1,1e-2;100,300 --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd22 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name  MC5ICL5unbalanced  --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd23 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name  MC5ICL5unbalanced  --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd24 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name  MC5ICL5unbalanced  --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 10 --n_iters 5 --unbalanced_data'

cmd25 = 'python  ./framework/main.py small_VGG9_cl_128_128 --gridsearch_name  MC5ICL5unbalanced  --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --num_epochs 10 --n_iters 5 --unbalanced_data'


# cmds = [cmd1]
# cmds = [cmd14, cmd15, cmd18, cmd19]
# cmds = [cmd2, cmd3, cmd4, cmd5, cmd7, cmd8, cmd10, cmd11, cmd14, cmd15, cmd16, cmd17, cmd22, cmd23, cmd24, cmd25, cmd18, cmd19]
cmds = [cmd10, cmd11, cmd22, cmd23, cmd24, cmd25]
for cmd in cmds:
    os.system(cmd)

