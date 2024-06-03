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

from loss import losses

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
            value,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs
        try:
            loss_function = losses[loss_fn]
        except:
            raise ValueError(f"the loss function type {loss_fn} is unknown. There exists {losses.keys()} loss functions.")

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        value = value.to(self.device)
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

        loss, nll, entropy = loss_function(
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
        pretraining=True,
        inner_epochs=10
    ):
        super().__init__(
            model=model, 
            optimizer=optimizer, 
            log_temperature_optimizer=log_temperature_optimizer,
            scheduler=scheduler, 
            device=device,
            pretraining=pretraining
        )
        
        self.inner_epochs = inner_epochs
        self.value_optimizer = value_optimizer

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
            loss, nll, entropy, value_loss = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            wandb.log({
                f"{self.pretraining}/train_loss": np.mean(loss),
                f"{self.pretraining}/nll": nll,
                f"{self.pretraining}/entropy": entropy,
                f"{self.pretraining}/temp_value": self.model.temperature().detach().cpu().item(),
                f"{self.pretraining}/value_loss": value_loss,
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
            values,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs
        
        try:
            loss_function = losses[loss_fn]
        except:
            raise ValueError(f"the loss function type {loss_fn} is unknown. There exists {losses.keys()} loss functions.")

        states = states.to(self.device) # (batchsize, sequence_length, state_dimension)
        actions = actions.to(self.device) # (batchsize, sequence_length, action_dimension)
        rewards = rewards.to(self.device) # (batchsize, sequence_length, 1)
        dones = dones.to(self.device) # (batchsize, sequence_length)
        rtg = rtg.to(self.device) # (batchsize, sequence_length, 1)
        values = values.to(self.device) # (batchsize, sequence_length, 1)
        timesteps = timesteps.to(self.device) # (batchsize, sequence_length)
        ordering = ordering.to(self.device) # (batchsize, sequence_length)
        padding_mask = padding_mask.to(self.device) # (batchsize, sequence_length)

        action_target = torch.clone(actions)

        state_preds, action_preds, return_preds, value_preds = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )
        
        loss, nll, entropy, value_loss = loss_function(
            model=self.model, 
            states=states, 
            actions=actions,
            rewards=rewards,
            rtg=rtg,
            timesteps=timesteps,
            ordering=ordering,
            padding_mask=padding_mask, 
            policy_optimizer=self.optimizer,
            value_target=values,
            value_optimizer=self.value_optimizer,
            inner_epochs=self.inner_epochs,
            clip_range=0.2,
            gamma=0.99
        )
        
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
            value_loss.detach().cpu().item()
        )