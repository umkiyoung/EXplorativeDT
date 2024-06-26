"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = "1"

import argparse
import pickle
import random
import time
import gym
import d4rl
import torch
import numpy as np
from tqdm import tqdm, trange
import wandb
import warnings
warnings.filterwarnings('ignore')


import utils
from replay_buffer import ReplayBuffer, create_dataloader
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from decision_transformer import DecisionTransformer, ValueDecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer

MAX_EPISODE_LEN = 1000


class Experiment:
    def __init__(self, variant):

        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        
        if variant['learning_from_offline_dataset']:
            self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
                variant["env"] ## DATA LEAKAGE IN CASE OF SCRATCH TRAINING
            )
        else: # If you learn from scratch, do not normalize the state.
            self.offline_trajs, self.state_mean, self.state_std = ([], np.array([0]), np.array([1]))
            if variant['num_updates_per_pretrain_iter'] != 0:
                raise ValueError('If you are trying to learn from the scratch, do not make the "num_updates_per_pretrain_iter" zero.')

        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)

        self.aug_trajs = []

        self.device = variant.get("device", "cuda")
        self.target_entropy = -self.act_dim
        self.model = ValueDecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
            gamma=variant['gamma']
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        
        self.value_optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        self.logger = utils.wandb_init(variant)
        self.tuning_type = 'Off policy' if self.variant["off_policy_tuning"] else "On policy"
        # if (self.variant["off_policy_tuning"]) and (self.variant['finetune_loss_fn']=='PPO'):
        #     raise ValueError(f"{self.tuning_type} and {self.variant['finetune_loss_fn']} are not compatible each other")

    def _get_env_spec(self, variant):
        env = gym.make(variant["env"])
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
        ]
        env.close()
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):  
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }
        folderpath = f"{path_prefix}/{self.variant['pretrain_loss_fn']}"
        # with open(f"{path_prefix}/model.pt", "wb") as f:
        #     torch.save(to_save, f)
        # print(f"\nModel saved at {path_prefix}/model.pt")
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        
        with open(f"{folderpath}/{self.variant['pretrain_loss_fn']}_{self.variant['env']}.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"Model saved at {folderpath}/{self.variant['pretrain_loss_fn']}_{self.variant['env']}.pt")

    def _load_model(self, path_prefix):
        folderpath = f"{path_prefix}/{self.variant['pretrain_loss_fn']}"
        
        if Path(f"{folderpath}/{self.variant['pretrain_loss_fn']}_{self.variant['env']}.pt").exists():
            with open(f"{folderpath}/{self.variant['pretrain_loss_fn']}_{self.variant['env']}.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {folderpath}/{self.variant['pretrain_loss_fn']}_{self.variant['env']}.pt")
        else:
            raise ValueError(f"There is no file at {folderpath}/{self.variant['pretrain_loss_fn']}_{self.variant['env']}.pt")

    def _load_dataset(self, env_name):

        dataset_path = f"./data/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs

            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
            )

        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def pretrain(self, eval_envs, loss_fn):
        

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            value_optimizer=self.value_optimizer,
            scheduler=self.scheduler,
            device=self.device,
            pretraining=True
        )
                
        
        
        if self.variant['num_updates_per_pretrain_iter'] > 0:
            print("\n\n\n*** Pretrain ***")
            print(f"Loss function configuration: {self.variant['pretrain_loss_fn']}")
            with tqdm(total=self.variant['max_pretrain_iters'], position=0, desc='Pretraining') as pbar:
                while self.pretrain_iter < self.variant["max_pretrain_iters"]:
                    # in every iteration, prepare the data loader
                    dataloader = create_dataloader(
                        trajectories=self.offline_trajs,
                        num_iters=self.variant["num_updates_per_pretrain_iter"],
                        batch_size=self.variant["batch_size"],
                        max_len=self.variant["K"],
                        state_dim=self.state_dim,
                        act_dim=self.act_dim,
                        state_mean=self.state_mean,
                        state_std=self.state_std,
                        reward_scale=self.reward_scale,
                        action_range=self.action_range,
                    )

                    train_outputs = trainer.train_iteration(
                        loss_fn=loss_fn,
                        dataloader=dataloader,
                    )
                    outputs = {"time/total": time.time() - self.start_time}
                    eval_outputs, eval_reward = self.evaluate(eval_fns)
                    outputs.update(eval_outputs)
                    outputs.update({'result/normalized_score': d4rl.get_normalized_score(self.variant['env'], eval_outputs['evaluation/return_mean_gm']) * 100})

                    outputs.update(train_outputs)
                    pbar.set_description(f"Pretraining | evaluation: {d4rl.get_normalized_score(self.variant['env'], eval_outputs['evaluation/return_mean_gm'] * 100):.1f}")
                    wandb.log(outputs, commit=False)

                    #self._save_model(
                    #     path_prefix=self.logger.log_path,
                    #     is_pretrain_model=True,
                    #)

                    self.pretrain_iter += 1
                    pbar.update(1)
        else:
            print("Skipped pretraining. starting scratch online training...")

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs, loss_fn):
        print("\n\n\n*** Online Finetuning ***")
        print(f"Loss function configuration: {self.variant['finetune_loss_fn']}")
        
        
        print(f"{self.tuning_type} training")
        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            value_optimizer=self.value_optimizer,
            scheduler=self.scheduler,
            device=self.device,
            pretraining=False,
            clip_range=self.variant["clip_range"],
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        with tqdm(total=self.variant["max_online_iters"], position=0, desc='Finetuning') as pbar:
            while self.online_iter < self.variant["max_online_iters"]:

                outputs = {}
                if self.tuning_type == 'On policy':
                    self.replay_buffer.reset()
                augment_outputs = self._augment_trajectories(
                    online_envs,
                    self.variant["online_rtg"],
                    n=self.variant["num_online_rollouts"],
                )
                outputs.update(augment_outputs)

                dataloader = create_dataloader(
                    trajectories=self.replay_buffer.trajectories,
                    num_iters=self.variant["num_updates_per_online_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    action_range=self.action_range,
                )

                # finetuning
                is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
                if (self.online_iter + 1) % self.variant[
                    "eval_interval"
                ] == 0 or is_last_iter:
                    evaluation = True
                else:
                    evaluation = False

                train_outputs = trainer.train_iteration(
                    loss_fn=loss_fn,
                    dataloader=dataloader,
                    finetuning_epoch=self.online_iter,
                    pretraining_epoch=self.variant["num_updates_per_pretrain_iter"]
                )
                outputs.update(train_outputs)

                if evaluation:
                    eval_outputs, eval_reward = self.evaluate(eval_fns)
                    outputs.update(eval_outputs)
                    outputs.update({'result/normalized_score': d4rl.get_normalized_score(self.variant['env'], eval_outputs['evaluation/return_mean_gm']) * 100})
                    pbar.set_description(f"Finetuning | evaluation: {d4rl.get_normalized_score(self.variant['env'], eval_outputs['evaluation/return_mean_gm'] * 100):.1f}")
                    
                outputs["time/total"] = time.time() - self.start_time
                trainer.update_behavioral_policy() # for PPO on policy learning.
                wandb.log(outputs, commit=False)
                
                # log the metrics
                # self.logger.log_metrics(
                #     outputs,
                #     iter_num=self.pretrain_iter + self.online_iter,
                #     total_transitions_sampled=self.total_transitions_sampled,
                #     writer=writer,
                # )

                # self._save_model(
                #     path_prefix=self.logger.log_path,
                #     is_pretrain_model=False,
                # )

                self.online_iter += 1
                pbar.update(1)


    def __call__(self):

        utils.set_seed_everywhere(args.seed)

        import d4rl

        pretrain_loss_fn = self.variant['pretrain_loss_fn']
        finetune_loss_fn = self.variant['finetune_loss_fn']

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():
                import d4rl

                env = gym.make(env_name)
                env.seed(seed)
                if hasattr(env.env, "wrapped_env"):
                    env.env.wrapped_env.seed(seed)
                elif hasattr(env.env, "seed"):
                    env.env.seed(seed)
                else:
                    pass
                env.action_space.seed(seed)
                env.observation_space.seed(seed)

                if target_goal:
                    env.set_target_goal(target_goal)
                    print(f"Set the target goal to be {env.target_goal}")
                return env

            return make_env_fn

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )

        self.start_time = time.time()
        if not self.variant["load_dir"]:
            # max_pretrain_iters determine the epochs of pretraining. If it is zero, skip the pretraining
            # If you want the offline data not to be loaded on replay buffer, set learning_from_offline_dataset False
            if self.variant["max_pretrain_iters"]: 
                self.pretrain(eval_envs, pretrain_loss_fn)
                if self.variant["save_dir"]:
                    self._save_model(self.variant['save_dir'], is_pretrain_model=True)
        else:   
            print("Skipped pretraining. starting online training with pretrained policy...")
            self._load_model(self.variant['load_dir'])

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            self.online_tuning(online_envs, eval_envs, finetune_loss_fn)
            online_envs.close()

        eval_envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # wandb logger
    parser.add_argument("--group", type=str, default='ExDT')
    parser.add_argument("--name", type=str, default='ExDT')
    parser.add_argument("--tag", type=str, default='')
    
    # ExDT hyperparameter
    parser.add_argument("--gamma", type=float, default=0.99)
    
    # seed and env
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    
    # loss function options
    parser.add_argument('--pretrain_loss_fn', type=str, default='ODT')
    parser.add_argument('--finetune_loss_fn', type=str, default='PPO')
    
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=3600)
    parser.add_argument("--num_eval_episodes", type=int, default=10)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--off_policy_tuning", default=False, action='store_true')
    parser.add_argument("--learning_from_offline_dataset", default=True, action='store_true') # if this is false, offline data is not adapted 
    parser.add_argument("--max_pretrain_iters", type=int, default=1) 
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=50)

    # finetuning options
    parser.add_argument("--load_from_pretrained_model", default=True, action='store_true')
    parser.add_argument("--max_online_iters", type=int, default=200)
    parser.add_argument("--max_online_steps", type=int, default=200000)
    parser.add_argument("--online_rtg", type=int, default=7200)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=30)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--clip_range", type=float, default=0.2)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    
    # save and load 
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--load_dir", type=str, default="")
    
    args = parser.parse_args()

    ## print current args:
    # print("Current args:")
    # for arg in vars(args):
    #     print(f"{arg}: {getattr(args, arg)}")
    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))

    print("=" * 50)
    experiment()