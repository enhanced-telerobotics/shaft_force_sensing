# Baselines
sbatch train_predict.sh --save_dir logs/transformer/base --lr 1e-3 --batch_size 256 --hidden_size 64 --model_type transformer --weight_decay 1e-4 --max_epochs 100
sbatch train_predict.sh --save_dir logs/lstm/base --lr 5e-3 --batch_size 256 --hidden_size 128 --model_type lstm --weight_decay 1e-6 --stride 5 --sequence_length 1000 --max_epochs 200
sbatch train_predict.sh --save_dir logs/fcn/base --lr 1e-3  --batch_size 256 --model_type fcn --weight_decay 1e-4 --sequence_length 1 --max_epochs 100

# Hyperparameter search
sbatch train_predict.sh --save_dir logs/lstm/ft_lr1e-4_wd1e-7 --max_epochs 200 --lr 1e-4 --weight_decay 1e-7 --sequence_length 1000 --stride 1 --model_type lstm --model_dir logs/lstm/base --finetune
sbatch train_predict.sh --save_dir logs/lstm/ft_lr1e-4_wd1e-6 --max_epochs 200 --lr 1e-4 --weight_decay 1e-6 --sequence_length 1000 --stride 1 --model_type lstm --model_dir logs/lstm/base --finetune # Best so far
sbatch train_predict.sh --save_dir logs/lstm/ft_lr1e-3_wd1e-6 --max_epochs 200 --lr 1e-3 --weight_decay 1e-6 --sequence_length 1000 --stride 1 --model_type lstm --model_dir logs/lstm/base --finetune

# Finetune with best hyperparameters from base training (selected)
# Transformer
sbatch train_predict.sh --save_dir logs/transformer/ft_m0 --max_epochs 100 --model_idx 0 --model_type transformer --lr 1e-3 --weight_decay 1e-4  --stride 5 --teleop --finetune --model_dir logs/transformer/base
sbatch train_predict.sh --save_dir logs/transformer/ft_m1 --max_epochs 100 --model_idx 1 --model_type transformer --lr 1e-3 --weight_decay 1e-4  --stride 5 --teleop --finetune --model_dir logs/transformer/base
sbatch train_predict.sh --save_dir logs/transformer/ft_m2 --max_epochs 100 --model_idx 2 --model_type transformer --lr 1e-3 --weight_decay 1e-4  --stride 5 --teleop --finetune --model_dir logs/transformer/base

# LSTM
sbatch train_predict.sh --save_dir logs/lstm/ft_m0 --max_epochs 200 --model_idx 0 --model_type lstm --lr 1e-4 --weight_decay 1e-6 --sequence_length 1000 --stride 5 --teleop --finetune --model_dir logs/lstm/base
sbatch train_predict.sh --save_dir logs/lstm/ft_m1 --max_epochs 200 --model_idx 1 --model_type lstm --lr 1e-4 --weight_decay 1e-6 --sequence_length 1000 --stride 5 --teleop --finetune --model_dir logs/lstm/base
sbatch train_predict.sh --save_dir logs/lstm/ft_m2 --max_epochs 200 --model_idx 2 --model_type lstm --lr 1e-4 --weight_decay 1e-6 --sequence_length 1000 --stride 5 --teleop --finetune --model_dir logs/lstm/base

# FCN
sbatch train_predict.sh --save_dir logs/fcn/ft_m0 --max_epochs 100 --model_idx 0 --model_type fcn --lr 1e-3 --weight_decay 1e-4  --stride 5 --sequence_length 1 --teleop --finetune --model_dir logs/fcn/base
sbatch train_predict.sh --save_dir logs/fcn/ft_m1 --max_epochs 100 --model_idx 1 --model_type fcn --lr 1e-3 --weight_decay 1e-4  --stride 5 --sequence_length 1 --teleop --finetune --model_dir logs/fcn/base
sbatch train_predict.sh --save_dir logs/fcn/ft_m2 --max_epochs 100 --model_idx 2 --model_type fcn --lr 1e-3 --weight_decay 1e-4  --stride 5 --sequence_length 1 --teleop --finetune --model_dir logs/fcn/base

# Scratch teleop (selected)
# Transformer
sbatch train_predict.sh --save_dir logs/transformer/scratch_m0 --max_epochs 100 --model_idx 0 --model_type transformer --lr 1e-3 --weight_decay 1e-4  --stride 5 --teleop
sbatch train_predict.sh --save_dir logs/transformer/scratch_m1 --max_epochs 100 --model_idx 1 --model_type transformer --lr 1e-3 --weight_decay 1e-4  --stride 5 --teleop
sbatch train_predict.sh --save_dir logs/transformer/scratch_m2 --max_epochs 100 --model_idx 2 --model_type transformer --lr 1e-3 --weight_decay 1e-4  --stride 5 --teleop

# LSTM
sbatch train_predict.sh --save_dir logs/lstm/scratch_m0 --max_epochs 200 --model_idx 0 --model_type lstm --lr 1e-3 --weight_decay 1e-6 --sequence_length 1000 --stride 5 --teleop
sbatch train_predict.sh --save_dir logs/lstm/scratch_m1 --max_epochs 200 --model_idx 1 --model_type lstm --lr 1e-3 --weight_decay 1e-6 --sequence_length 1000 --stride 5 --teleop
sbatch train_predict.sh --save_dir logs/lstm/scratch_m2 --max_epochs 200 --model_idx 2 --model_type lstm --lr 1e-3 --weight_decay 1e-6 --sequence_length 1000 --stride 5 --teleop

# FCN
sbatch train_predict.sh --save_dir logs/fcn/scratch_m0 --max_epochs 100 --model_idx 0 --model_type fcn --lr 1e-3 --weight_decay 1e-4  --stride 5 --sequence_length 1 --teleop
sbatch train_predict.sh --save_dir logs/fcn/scratch_m1 --max_epochs 100 --model_idx 1 --model_type fcn --lr 1e-3 --weight_decay 1e-4  --stride 5 --sequence_length 1 --teleop
sbatch train_predict.sh --save_dir logs/fcn/scratch_m2 --max_epochs 100 --model_idx 2 --model_type fcn --lr 1e-3 --weight_decay 1e-4  --stride 5 --sequence_length 1 --teleop
