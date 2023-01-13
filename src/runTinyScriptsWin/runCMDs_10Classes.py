import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cmd1= 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes --method_name  joint  --ds_name tiny   --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'
cmd2 = 'python ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes --method_name  joint  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50 --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'


cmds = [cmd1, cmd2]
for cmd in cmds:
    os.system(cmd)

