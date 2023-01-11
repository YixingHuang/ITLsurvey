import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
cmd1 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name optimizer_Adam_STCL_redo_lambda1 --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --num_epochs 50 --no_maximal_plasticity_search'  # --ini_path first_task_basemodel_ICL2 --stochastic

cmd18 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name code_restructure --method_name SI --ds_name tiny  --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --hyperparams 1'
cmd19 = 'python  ./framework/main.py  small_VGG9_cl_128_128  --gridsearch_name code_restructure --method_name SI --ds_name tiny --lr_grid 1e-2,5e-3,1e-3,5e-4,1e-4 --boot_lr_grid 1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7 --no_maximal_plasticity_search --first_task_basemodel_folder first_task_basemodel --hyperparams 1'


# cmds = [cmd1]
# cmds = [cmd20, cmd21, cmd18, cmd19]
# cmds = [cmd1, cmd18, cmd19, cmd22, cmd23, cmd24, cmd25, cmd26, cmd27, cmd28, cmd29]
cmds = [cmd18]
for cmd in cmds:
    os.system(cmd)

