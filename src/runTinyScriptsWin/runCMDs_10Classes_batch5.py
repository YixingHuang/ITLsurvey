import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cmd1= 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  joint  --ds_name tiny   --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'
cmd2 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  joint  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'

cmd0 = 'py  ../framework/main.py small_VGG9_cl_128_128  --runmode first_task_basemodel_dump --gridsearch_name Adam_10Classes_batch5 --method_name SI --ds_name tiny   --num_epochs 50  --batch_size 5    --no_maximal_plasticity_search   --first_task_basemodel_folder first_task_Adam_10Classes_batch5 --optimizer 1  --fixed_init_lr 0.001  --num_class 10'  # --ini_path first_task_basemodel_ICL2 --stochastic

cmd13 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  FT  --ds_name tiny   --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'
cmd14 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  FT  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'

cmd3 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  EWC  --ds_name tiny   --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'
cmd4 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  EWC  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'

cmd5 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  LWF  --ds_name tiny   --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'
cmd6 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  LWF  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'

cmd7 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  EBLL  --ds_name tiny   --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --static_hyperparams 0.01,0.001;50;1e-1,1e-2;100,300'
cmd8 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  EBLL  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 --static_hyperparams 0.01,0.001;50;1e-1,1e-2;100,300'

cmd9 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  MAS  --ds_name tiny   --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 '
cmd10 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  MAS --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10 '

cmd11 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  SI  --ds_name tiny   --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'
cmd12 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  SI  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'


cmd15 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  meanIMM  --ds_name tiny   --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'
cmd16 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  meanIMM  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'

cmd17 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  modeIMM  --ds_name tiny   --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'
cmd18 = 'py ../framework/main.py small_VGG9_cl_128_128  --gridsearch_name Adam_10Classes_batch5 --method_name  modeIMM  --ds_name tiny   --test  --test_overwrite_mode --num_epochs 50  --batch_size 5  --n_iters 1 --seed 7   --no_maximal_plasticity_search  --first_task_basemodel_folder first_task_Adam_10Classes_batch5    --optimizer 1  --fixed_init_lr 0.001 --num_class 10'

# cmds = [cmd1, cmd2, cmd0, cmd11, cmd12, cmd3, cmd4, cmd5, cmd6, cmd9, cmd10, cmd7, cmd8]
cmds = [cmd17, cmd18]
for cmd in cmds:
    os.system(cmd)

