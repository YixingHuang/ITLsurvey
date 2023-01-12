import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


cmd1 = 'py ./framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name Adam_200Classes --method_name SI --ds_name tiny   --num_epochs 50   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes --optimizer 1  --fixed_init_lr 0.001  --num_class 200'  # --ini_path first_task_basemodel_ICL2 --stochastic

# Adam  FT
cmd20 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd21 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

#Adam SI
cmd30 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd31 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

# Adam IMM
cmd60 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  meanIMM  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd61 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  meanIMM  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

# Adam Joint
cmd162 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  joint  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd163 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  joint  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

# Adam EWC
cmd164 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  EWC  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd165 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

# Adam MAS
cmd166 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  MAS  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd167 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  MAS  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

# Adam LWF
cmd168 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  LWF  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd169 = 'py ./framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_200Classes --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

###########################base_VGG9_cl_512_512
##########################
cmd2 = 'py ./framework/main.py base_VGG9_cl_512_512  --runmode first_task_basemodel_dump --gridsearch_name Adam_200Classes --method_name SI --ds_name tiny   --num_epochs 50   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes --optimizer 1  --fixed_init_lr 0.001  --num_class 200'  # --ini_path first_task_basemodel_ICL2 --stochastic

# Adam  FT
cmd22 = 'py ./framework/main.py base_VGG9_cl_512_512  --gridsearch_name Adam_200Classes --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd23 = 'py ./framework/main.py base_VGG9_cl_512_512  --gridsearch_name Adam_200Classes --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

#Adam SI
cmd32 = 'py ./framework/main.py base_VGG9_cl_512_512  --gridsearch_name Adam_200Classes --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd33 = 'py ./framework/main.py base_VGG9_cl_512_512  --gridsearch_name Adam_200Classes --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '


# Adam IMM
cmd62 = 'py ./framework/main.py base_VGG9_cl_512_512  --gridsearch_name Adam_200Classes --method_name  meanIMM  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd63 = 'py ./framework/main.py base_VGG9_cl_512_512  --gridsearch_name Adam_200Classes --method_name  meanIMM  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

###########################base_VGG9_cl_512_512
##########################
cmd3 = 'py ./framework/main.py deep_VGG22_cl_512_512  --runmode first_task_basemodel_dump --gridsearch_name Adam_200Classes --method_name SI --ds_name tiny   --num_epochs 50   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes --optimizer 1  --fixed_init_lr 0.001  --num_class 200'  # --ini_path first_task_basemodel_ICL2 --stochastic

# Adam  FT
cmd24 = 'py ./framework/main.py deep_VGG22_cl_512_512  --gridsearch_name Adam_200Classes --method_name  FT  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd25 = 'py ./framework/main.py deep_VGG22_cl_512_512  --gridsearch_name Adam_200Classes --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

#Adam SI
cmd34 = 'py ./framework/main.py deep_VGG22_cl_512_512  --gridsearch_name Adam_200Classes --method_name  SI  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd35 = 'py ./framework/main.py deep_VGG22_cl_512_512  --gridsearch_name Adam_200Classes --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '


# Adam IMM
cmd64 = 'py ./framework/main.py deep_VGG22_cl_512_512  --gridsearch_name Adam_200Classes --method_name  meanIMM  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '
cmd65 = 'py ./framework/main.py deep_VGG22_cl_512_512  --gridsearch_name Adam_200Classes --method_name  meanIMM  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_200Classes    --optimizer 1  --fixed_init_lr 0.001   '

# cmds = [cmd61]
# cmd1, cmd20, cmd21, cmd30, cmd31,
# cmds = [cmd60, cmd61, cmd2, cmd22, cmd23, cmd32, cmd33, cmd62, cmd63, cmd3, cmd24, cmd25, cmd34, cmd35, cmd64, cmd65]
# cmds = [cmd162, cmd163, cmd164, cmd165, cmd166, cmd167, cmd168, cmd169]
cmds = [cmd167]
for cmd in cmds:
    os.system(cmd)

