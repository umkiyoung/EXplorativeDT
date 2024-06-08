device=cuda:1

python ./ExDT.py \
    --env=halfcheetah-medium-v2 \
    --tag=$tag \
    --pretrain_loss_fn=ODT \
    --finetune_loss_fn=ODT \
    --learning_from_offline_dataset \
    --device=$device  \
    --num_updates_per_pretrain_iter=5000 \
    --learning_rate=0.0001 \
    --weight_decay=0.0005 \
    --eval_context_length=5 \
    --eval_rtg=6000 \
    --online_rtg=12000 \
    --num_updates_per_online_iter=300 \
    --off_policy_tuning \
    --save_dir=pretrained_policy &


python ./ExDT.py \
    --env=halfcheetah-medium-replay-v2 \
    --tag=$tag \
    --pretrain_loss_fn=ODT \
    --finetune_loss_fn=ODT \
    --learning_from_offline_dataset \
    --device=$device  \
    --num_updates_per_pretrain_iter=5000 \
    --learning_rate=0.0001 \
    --weight_decay=0.0005 \
    --eval_context_length=5 \
    --eval_rtg=6000 \
    --online_rtg=12000 \
    --num_updates_per_online_iter=300 \
    --off_policy_tuning \
    --save_dir=pretrained_policy 


python ./ExDT.py \
    --env=hopper-medium-v2 \
    --tag=$tag \
    --pretrain_loss_fn=ODT \
    --finetune_loss_fn=ODT \
    --learning_from_offline_dataset \
    --device=$device  \
    --num_updates_per_pretrain_iter=5000 \
    --learning_rate=0.0001 \
    --weight_decay=0.0005 \
    --eval_context_length=5 \
    --eval_rtg=3600 \
    --online_rtg=7200 \
    --num_updates_per_online_iter=300 \
    --off_policy_tuning \
    --save_dir=pretrained_policy &


python ./ExDT.py \
    --env=hopper-medium-replay-v2 \
    --tag=$tag \
    --pretrain_loss_fn=ODT \
    --finetune_loss_fn=ODT \
    --learning_from_offline_dataset \
    --device=$device  \
    --num_updates_per_pretrain_iter=5000 \
    --learning_rate=0.002 \
    --weight_decay=0.0001 \
    --eval_context_length=5 \
    --eval_rtg=3600 \
    --online_rtg=7200 \
    --num_updates_per_online_iter=300 \
    --off_policy_tuning \
    --save_dir=pretrained_policy 


python ./ExDT.py \
    --env=walker2d-medium-v2 \
    --tag=$tag \
    --pretrain_loss_fn=ODT \
    --finetune_loss_fn=ODT \
    --learning_from_offline_dataset \
    --device=$device  \
    --num_updates_per_pretrain_iter=10000 \
    --learning_rate=0.001 \
    --weight_decay=0.001 \
    --eval_context_length=5 \
    --eval_rtg=5000 \
    --online_rtg=10000 \
    --num_updates_per_online_iter=300 \
    --off_policy_tuning \
    --save_dir=pretrained_policy &


python ./ExDT.py \
    --env=walker2d-medium-replay-v2 \
    --tag=$tag \
    --pretrain_loss_fn=ODT \
    --finetune_loss_fn=ODT \
    --learning_from_offline_dataset \
    --device=$device  \
    --num_updates_per_pretrain_iter=10000 \
    --learning_rate=0.001 \
    --weight_decay=0.001 \
    --eval_context_length=5 \
    --eval_rtg=5000 \
    --online_rtg=10000 \
    --num_updates_per_online_iter=300 \
    --off_policy_tuning \
    --save_dir=pretrained_policy \ 

