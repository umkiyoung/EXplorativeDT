device=cuda:0
python ./ExDT.py \
    --tag=$tag \
    --pretrain_loss_fn=ODT \
    --finetune_loss_fn=ODT \
    --learning_from_offline_dataset \
    --device=$device  \
    --env=halfcheetah-medium-v2 \
    --num_updates_per_pretrain_iter=5000 \
    --learning_rate=0.0001 \
    --weight_decay=0.0005 \
    --eval_context_length=5 \
    --eval_rtg=6000 \
    --online_rtg=12000 \
    --num_updates_per_online_iter=300 \
    --off_policy_tuning 