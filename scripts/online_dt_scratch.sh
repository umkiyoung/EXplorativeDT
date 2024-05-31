device=cuda:0

python /home/jaewoo/research/EXplorativeDT/main.py --device=$device --env=hopper-medium-v2 --num_updates_per_pretrain_iter=0 --learning_rate=0.0001 --weight_decay=0.0005 --eval_context_length=5 --eval_rtg=3600 --online_rtg=7200
python /home/jaewoo/research/EXplorativeDT/main.py --device=$device --env=hopper-medium-replay-v2 --learning_rate=0.002 --num_updates_per_pretrain_iter=0 --weight_decay=0.0001 --eval_context_length=5 --eval_rtg=3600 --online_rtg=7200

python /home/jaewoo/research/EXplorativeDT/main.py --device=$device --env=walker2d-medium--v2 --learning_rate=0.001 --num_updates_per_pretrain_iter=0 --weight_decay=0.001 --eval_context_length=5 --eval_rtg=0 --online_rtg=10000
python /home/jaewoo/research/EXplorativeDT/main.py --device=$device --env=walker2d-medium-replay-v2 --learning_rate=0.001 --num_updates_per_pretrain_iter=0 --weight_decay=0.001 --eval_context_length=5 --eval_rtg=0 --online_rtg=10000

python /home/jaewoo/research/EXplorativeDT/main.py --device=$device --env=halfcheetah-medium-v2 --num_updates_per_pretrain_iter=0 --learning_rate=0.0001 --weight_decay=0.0005 --eval_context_length=5 --eval_rtg=6000 --online_rtg=12000
python /home/jaewoo/research/EXplorativeDT/main.py --device=$device --env=halfcheetah-medium-replay-v2 --num_updates_per_pretrain_iter=0 --learning_rate=0.0001 --weight_decay=0.0005 --eval_context_length=5 --eval_rtg=6000 --online_rtg=12000