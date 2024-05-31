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

class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
        pretraining=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()
        if pretraining:
            self.pretraining = "pretraining"
        else:
            self.pretraining = 'finetuning'

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
            loss, nll, entropy = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            wandb.log({
                f"{self.pretraining}/train_loss": np.mean(loss),
                f"{self.pretraining}/nll": nll,
                f"{self.pretraining}/entropy": entropy,
                f"{self.pretraining}/temp_value": self.model.temperature().detach().cpu().item(),
                },
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
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )

class ValueSequenceTrainer(SequenceTrainer):
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        value_optimizer,
        scheduler=None,
        device="cuda",
    ):
        super().__init__(model, optimizer, log_temperature_optimizer, value_optimizer, scheduler, device)

    def train_iteration(
        self,
        loss_fn,
        dataloader,
    ):

        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _, trajs in enumerate(dataloader):
            loss, nll, entropy = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs

    def train_step_stochastic(self, loss_fn, value_loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            values, # added
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        values = values.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)
        value_target = torch.clone(values)

        _, action_preds, _, value_preds = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )
        
        td_loss = value_loss_fn( # add value loss function
            value_preds,
            value_target
        )
        
        total_loss = loss + td_loss
        
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        self.value_optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )