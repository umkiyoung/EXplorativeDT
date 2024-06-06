"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
from tqdm import tqdm
import wandb
import time
import numpy as np  # Add missing import statement

from loss import get_loss_function


class SequenceTrainer():
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
        pretraining=False,
        clip_range=1e-4,
        value_optimizer=None,
        inner_epochs=10,
    ):
        self.model = model
        self.policy_optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()
        
        if pretraining:
            self.pretraining = "pretraining"
        else:
            self.pretraining = 'finetuning'
        
        self.clip_range = clip_range
        self.value_optimizer = value_optimizer
        self.inner_epochs = inner_epochs
            

        
    def train_iteration(
        self,
        loss_fn,
        dataloader,
        finetuning_epoch=0,
        pretraining_epoch=0,
    ):  

        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()
        self.model.train()
        for idx, trajs in enumerate(tqdm(dataloader, position=1, leave=False)):
            loss, nll, entropy, value = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            log_data = {
                f"{self.pretraining}/train_loss": np.mean(loss),
                f"{self.pretraining}/nll": nll,
                f"{self.pretraining}/entropy": entropy,
                f"{self.pretraining}/temp_value": self.model.temperature().detach().cpu().item(),
            }

            if value is not None:
                log_data[f"{self.pretraining}/value"] = value

            wandb.log(
                log_data,
                step=(idx + dataloader.__len__() * finetuning_epoch + pretraining_epoch),
                commit=False
            )

        logs["time/training"] = time.time() - train_start
        logs[f"{self.pretraining}/train_loss_mean"] = np.mean(losses)
        logs[f"{self.pretraining}/train_loss_std"] = np.std(losses)
        #logs["training/nll"] = nlls[-1]
        #logs["training/entropy"] = entropies[-1]
        #logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs

    def train_step_stochastic(self, loss_fn, trajs):
        trajs = [traj.to(self.device) for traj in trajs]
        states, actions, rewards, dones, rtg, values, timesteps, ordering, padding_mask = trajs
        Loss_Class = get_loss_function(loss_fn)
        if loss_fn in ["ODT","TRPO"]:
            loss, nll, entropy = Loss_Class.compute_loss(
                model=self.model, 
                states=states, 
                actions=actions,
                rewards=rewards,
                rtg=rtg,
                timesteps=timesteps,
                ordering=ordering,
                padding_mask=padding_mask, 
                policy_optimizer=self.policy_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
            )
            return loss, nll, entropy, None
            
        elif loss_fn in ["PPO", "ExDT"]:
            loss, nll, entropy, value = Loss_Class.compute_loss(
                model=self.model, 
                states=states, 
                actions=actions,
                rewards=rewards,
                rtg=rtg,
                timesteps=timesteps,
                ordering=ordering,
                padding_mask=padding_mask, 
                policy_optimizer=self.policy_optimizer,
                value_target=values,
                value_optimizer=self.value_optimizer,
                clip_range=self.clip_range,
                inner_epochs=self.inner_epochs,
                entropy_reg=self.model.temperature().detach(),
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
            )
            return loss, nll, entropy, value