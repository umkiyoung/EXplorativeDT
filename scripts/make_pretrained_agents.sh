device=cuda:0

python ExDT.py --save_dir=pretrained_policy --off_policy_tuning --finetune_loss_fn=ODT --device=$device --env=walker2d-medium-replay-v2 --learning_rate=0.001 --num_updates_per_pretrain_iter=10000 --weight_decay=0.001 --eval_context_length=5 --eval_rtg=5000 --online_rtg=10000&
python ExDT.py --save_dir=pretrained_policy --off_policy_tuning --finetune_loss_fn=ODT --device=$device --env=halfcheetah-medium-v2 --num_updates_per_pretrain_iter=5000 --learning_rate=0.0001 --weight_decay=0.0005 --eval_context_length=5 --eval_rtg=6000 --online_rtg=12000&
python ExDT.py --save_dir=pretrained_policy --off_policy_tuning --finetune_loss_fn=ODT --device=$device --env=halfcheetah-medium-replay-v2 --num_updates_per_pretrain_iter=5000 --learning_rate=0.0001 --weight_decay=0.0005 --eval_context_length=5 --eval_rtg=6000 --online_rtg=12000&

wait

python ExDT.py --save_dir=pretrained_policy --off_policy_tuning --finetune_loss_fn=ODT --device=$device --env=hopper-medium-v2 --num_updates_per_pretrain_iter=5000 --learning_rate=0.0001 --weight_decay=0.0005 --eval_context_length=5 --eval_rtg=3600 --online_rtg=7200 & 
python ExDT.py --save_dir=pretrained_policy --off_policy_tuning --finetune_loss_fn=ODT --device=$device --env=hopper-medium-replay-v2 --learning_rate=0.002 --num_updates_per_pretrain_iter=5000 --weight_decay=0.0001 --eval_context_length=5 --eval_rtg=3600 --online_rtg=7200 & 
python ExDT.py --save_dir=pretrained_policy --off_policy_tuning --finetune_loss_fn=ODT --device=$device --env=walker2d-medium-v2 --learning_rate=0.001 --num_updates_per_pretrain_iter=10000 --weight_decay=0.001 --eval_context_length=5 --eval_rtg=5000 --online_rtg=10000