import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
cmd1 = 'py  ./framework/main.py base_VGG9_cl_512_512 --runmode first_task_basemodel_dump --gridsearch_name reproduce --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4'
cmd2= ' py  ./framework/main.py base_VGG9_cl_512_512 --gridsearch_name reproduce --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4'

cmd3 = ' py  ./framework/main.py base_VGG9_cl_512_512 --gridsearch_name reproduce --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode'

# cmd3 = ' py  ./utilities/plot_configs/demo.py'

cmd4 = ' py  ./framework/main.py --runmode --gridsearch_name reproduce --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 small_VGG9_cl_128_128'

cmd5 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name MAS --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test base_VGG9_cl_512_512 --test_overwrite_mode'

# cmd6 = ' py  ./utilities/plot_configs/demoMAS.py'

cmd7 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 small_VGG9_cl_128_128'

cmd8 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name LWF --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test base_VGG9_cl_512_512 --test_overwrite_mode '

# cmd9 = ' py  ./utilities/plot_configs/demoLWF.py'

cmd10 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 base_VGG9_cl_512_512'

cmd11 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name EWC --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test base_VGG9_cl_512_512 --test_overwrite_mode '

# cmd12 = ' py  ./utilities/plot_configs/demoEWC.py'
#
# cmd13 = ' py  ./utilities/plot_configs/demoCombined.py'
# cmd13 = ' py  ./framework/main.py --runmode first_task_basemodel_dump --gridsearch_name reproduce --method_name EBLL --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 base_VGG9_cl_512_512'
#
# cmd14 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name EBLL --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test base_VGG9_cl_512_512 --test_overwrite_mode '
#
#
cmd14 = ' py  ./framework/main.py base_VGG9_cl_512_512 --gridsearch_name reproduce --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4'

cmd15 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name Fine_tuning --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test base_VGG9_cl_512_512 --test_overwrite_mode '

cmd16 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 base_VGG9_cl_512_512'

cmd17 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name joint --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test base_VGG9_cl_512_512 --test_overwrite_mode --test_max_task_count 10 '

cmd18 = 'py  ./framework/main.py base_VGG9_cl_512_512 --gridsearch_name reproduce --method_name EBLL --ds_name tiny   --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4  --static_hyperparams 0.01,0.001;50;1e-1,1e-2;100,300'

cmd19 = 'py  ./framework/main.py base_VGG9_cl_512_512 --gridsearch_name reproduce --method_name EBLL --ds_name tiny   --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test --test_overwrite_mode --static_hyperparams 0.01,0.001;50;1e-1,1e-2;100,300'

cmd20 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name packnet --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 base_VGG9_cl_512_512'

cmd21 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name packnet --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test base_VGG9_cl_512_512 --test_overwrite_mode'

cmd22 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 base_VGG9_cl_512_512'

cmd23 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name modeIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test base_VGG9_cl_512_512 --test_overwrite_mode '

cmd24 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 base_VGG9_cl_512_512'

cmd25 = ' py  ./framework/main.py --gridsearch_name reproduce --method_name meanIMM --ds_name tiny    --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test base_VGG9_cl_512_512 --test_overwrite_mode '
cmd30 = 'py  ./utilities/plot_configs/demoEBLLBase.py'
# os.system(cmd3)

# cmds = [cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd10, cmd11, cmd12]
# cmds = [cmd20, cmd21]
# cmds = [cmd1, cmd2, cmd3]
# cmds = [cmd18, cmd19, cmd20, cmd21, cmd22, cmd23, cmd24, cmd25]
# cmds = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd7, cmd8, cmd10, cmd11, cmd14, cmd15, cmd16, cmd17, cmd18, cmd19, cmd20, cmd21, cmd22, cmd23, cmd24, cmd25]

cmds = [cmd18, cmd19, cmd30]
for cmd in cmds:
    os.system(cmd)

