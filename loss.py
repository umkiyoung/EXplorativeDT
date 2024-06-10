from collections import defaultdict
from tqdm import trange, tqdm
import torch
import numpy as np
import wandb

import abc

class LossAbstract(abc.ABC):    
    def __init__(self, pretraining):
        self.pretraining=pretraining
    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass

class ODTLoss(LossAbstract):
    def loss_function(
        self,
        a_hat_dist,
        a,
        attention_mask,
        entropy_reg,
    ):
        # a_hat is a SquashedNormal Distribution
        log_likelihood = a_hat_dist.log_likelihood(a)[:, :-1][attention_mask[:, :-1] > 0].mean()
        entropy = a_hat_dist.entropy().mean()
        loss = -(log_likelihood + entropy_reg * entropy)
        return (
            loss,
            -log_likelihood,
            entropy,
        )
        
    def compute_loss(
            self,
            model, 
            states, 
            actions,
            rewards,
            rtg,
            timesteps,
            ordering,
            padding_mask, 
            policy_optimizer,
            log_temperature_optimizer,
            scheduler=None,
        ):
        
        action_target = actions.clone()
        
        _, action_preds, _, _ = model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )
        
        loss, nll, entropy = self.loss_function(
            action_preds,
            action_target,
            padding_mask,
            model.temperature().detach(),
        )
        
        policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        policy_optimizer.step()
        
        log_temperature_optimizer.zero_grad()
        temperature_loss = (
            model.temperature() * (entropy - model.target_entropy).detach()
        )
        temperature_loss.backward()
        log_temperature_optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
            
        info = {
            f"{self.pretraining}/train_loss": np.mean(loss.detach().cpu().item()),
            f"{self.pretraining}/nll": nll.detach().cpu().item(),
            f"{self.pretraining}/entropy": entropy.detach().cpu().item(),
            f"{self.pretraining}/temp_value": model.temperature().detach().cpu().item(),
        }
        
        wandb.log(info, commit=True)
                
        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )
        
class ODTValueLoss(LossAbstract):
    def loss_function(
        self,
        a_hat_dist,
        a,
        attention_mask,
        entropy_reg,
    ):
        # a_hat is a SquashedNormal Distribution
        log_likelihood = a_hat_dist.log_likelihood(a)[:, :-1][attention_mask[:, :-1] > 0].mean()
        entropy = a_hat_dist.entropy().mean()
        loss = -(log_likelihood + entropy_reg * entropy)
        return (
            loss,
            -log_likelihood,
            entropy,
        )
        
    def compute_loss(
        self,
        model, 
        states, 
        actions,
        rewards,
        rtg,
        timesteps,
        ordering,
        padding_mask, 
        policy_optimizer,
        value_target,
        log_temperature_optimizer,
        scheduler=None,
        max_grad_norm = 0.5
        ):
        
        action_target = actions.clone()
        
        _, action_preds, _, value_preds = model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )
        
        
        loss, nll, entropy = self.loss_function(
            action_preds,
            action_target,
            padding_mask,
            model.temperature().detach(),
        )
        
        value_loss = torch.nn.functional.mse_loss(value_preds[padding_mask > 0], value_target.clone()[:, :-1, :][padding_mask > 0]) # Value target starts from 1 to make TD loss
        loss = loss + 0.5 * value_loss
        policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        policy_optimizer.step()
        
        log_temperature_optimizer.zero_grad()
        temperature_loss = (
            model.temperature() * (entropy - model.target_entropy).detach()
        )
        temperature_loss.backward()
        log_temperature_optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
            
        info = {
            f"{self.pretraining}/train_loss": np.mean(loss.detach().cpu().item()),
            f"{self.pretraining}/nll": nll.detach().cpu().item(),
            f"{self.pretraining}/entropy": entropy.detach().cpu().item(),
            f"{self.pretraining}/temp_value": model.temperature().detach().cpu().item(),
        }
        
        wandb.log(info, commit=True)
                
        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )
        
        

class TRPOLoss(LossAbstract):
    def loss_function(
        self,
        a_hat_dist, # action distribution of decision transformer
        a, # action from replay buffer
        values, # for advantage
        attention_mask,
        bc_reg,
        inner_epochs
    ):
        # before the inner epoch loop, each action must have the log_prob of the old policy
        # log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
        behavioral_log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

        for inner_epoch in trange(inner_epochs, desc='TRPO loss propagation', position=2, leave=False):
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
            advantage = None # need to be implemented
            ratio = torch.exp(log_likelihood - behavioral_log_likelihood)
            bc_loss = -bc_reg * log_likelihood
            entropy = a_hat_dist.entropy().mean()
            loss = - advantage * ratio - bc_reg * log_likelihood # TRPO loss with dual gradient descent. 

            ## TODO if you want to implement TRPO logic, optimizer step must be in the inner loop.
            ## bc_reg must be tuned in here.
            log_temperature_optimizer.zero_grad()
            temperature_loss = (
                model.temperature() * (entropy - self.model.target_entropy).detach()
            )
            temperature_loss.backward()
            log_temperature_optimizer.step()
            return (
                loss,
                -log_likelihood,
                entropy,
            )
            
    def compute_loss(
        self,
        model, 
        states, 
        actions,
        rewards,
        rtg,
        timesteps,
        ordering,
        padding_mask, 
        policy_optimizer,
    ):
        action_target = actions.clone()
        
        _, action_preds, _ = model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )
        
        loss, nll, entropy = self.loss_function(
            action_preds,
            action_target,
            padding_mask,
            model.temperature().detach(),
        )
        
        policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        policy_optimizer.step()
        
        model.train()
        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )

class PPOLoss(LossAbstract):
    def compute_loss(
        self,
        model, 
        behavioral_model,
        states, 
        actions,
        rewards,
        rtg,
        timesteps,
        ordering,
        padding_mask, 
        policy_optimizer,
        value_target,
        value_optimizer,
        inner_epochs,
        clip_range,
        entropy_reg,
        log_temperature_optimizer,
        scheduler=None,
        gamma=0.99,
        max_grad_norm = 0.5
        ):
        # state_preds, action_preds, return_preds, value_preds = model.forward(
        #     states,
        #     actions,
        #     rewards,
        #     rtg[:, :-1],
        #     timesteps,
        #     ordering,
        #     padding_mask=padding_mask,
        # )
                
        # Policy Evaluation
        # value_loss = torch.nn.functional.mse_loss(value_preds[padding_mask > 0], value_target.clone()[:, :-1, :][padding_mask > 0]) # Value target starts from 1 to make TD loss
        # value_optimizer.zero_grad()
        # value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm) #grad norm add for spike prevention
        # value_optimizer.step()
        action_target = actions.clone()
        
        # Policy Improvement
        # before the inner epoch loop, each action must have the log_prob of the old policy
        # Turn off the dropout layer to stabilize the PPO ratio calculation
        # If there are no eval(), the ratio goes to inf
        behavioral_model.eval() 
        
        state_preds, action_preds, return_preds, value_preds = behavioral_model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )
        
        with torch.no_grad():
            behavioral_log_likelihood = action_preds.log_likelihood(action_target)[:, :-1][padding_mask[:, :-1] > 0].detach().clone()

        for inner_epoch in tqdm(range(inner_epochs), desc='PPO loss propagation', position=2, leave=False):
            state_preds, action_preds, return_preds, value_preds = model.forward(
                states,
                actions,
                rewards,
                rtg[:, :-1],
                timesteps,
                ordering,
                padding_mask=padding_mask,
            )
            
            value_loss = torch.nn.functional.mse_loss(value_preds[padding_mask > 0], value_target.clone()[:, :-1, :][padding_mask > 0]) # Value target starts from 1 to make TD loss
            
            log_likelihood = action_preds.log_likelihood(action_target)[:, :-1][padding_mask[:, :-1] > 0]
            current_step_rewards = rewards[:, :-1, :]
            current_step_values = value_preds.detach().clone()[:, :-1, :] # make it sure to detach the grad graph
            next_step_values = value_preds.detach().clone()[:, 1:, :] # (batch_size, seq_len, )
            advantage = current_step_rewards + gamma * next_step_values - current_step_values 
            advantage = (advantage - advantage.mean()) / advantage.std()
            ratio = torch.exp((log_likelihood.mean() - behavioral_log_likelihood.mean()).clamp_(max=5)) # preventing nan exploding
            unclipped_loss = -advantage * ratio # adv weighted likelihood without last step
            clipped_loss = -advantage * torch.clamp(
                ratio,
                1. - clip_range,
                1. + clip_range
            )
            entropy = action_preds.entropy().mean()
            ppo_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
            total_loss = ppo_loss +  0.5 * value_loss # - entropy_reg * entropy
            
            ## TODO which is better? integrated loss or eval - improve iteration?
            
            policy_optimizer.zero_grad()
            total_loss.backward() ## TODO is this the optimal solution?
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm) #grad norm add for spike prevention
            policy_optimizer.step()
            
        
            info = {
                f"{self.pretraining}/train_loss": np.mean(total_loss.detach().cpu().item()),
                f"{self.pretraining}/ppo_loss": ppo_loss.mean().detach().cpu().item(),
                f"{self.pretraining}/value_loss": value_loss.mean().detach().cpu().item(),
                f"{self.pretraining}/clipfrac": torch.mean((torch.abs(ratio-1.) > clip_range).float()),
                f"{self.pretraining}/approx_kl": 0.5 * torch.mean((log_likelihood.mean() - behavioral_log_likelihood.mean()) ** 2),
                f"{self.pretraining}/nll": -log_likelihood.mean().detach().cpu().item(),
                f"{self.pretraining}/entropy": entropy.detach().cpu().item(),
                f"{self.pretraining}/temp_value": model.temperature().detach().cpu().item(),
                f"{self.pretraining}/advantage_mean": (current_step_rewards + gamma * next_step_values - current_step_values).mean().detach().cpu().item(),
                f"{self.pretraining}/advantage_std": (current_step_rewards + gamma * next_step_values - current_step_values).std().detach().cpu().item(),
                f"{self.pretraining}/value_target_mean": value_target.mean().detach().cpu().item(),
                f"{self.pretraining}/value_pred_mean": value_preds.mean().detach().cpu().item(),
                f"{self.pretraining}/value_pred_std": value_preds.std().detach().cpu().item(),
                f"{self.pretraining}/reward_mean": rewards.mean().detach().cpu().item(),
                f"{self.pretraining}/rtg_mean": rtg.mean().detach().cpu().item()
            }
            
            wandb.log(info, commit=True)
            
            
        log_temperature_optimizer.zero_grad()
        temperature_loss = (
            model.temperature() * (entropy - model.target_entropy).detach()
        )
        temperature_loss.backward()
        log_temperature_optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        model.train()
        return (
            ppo_loss.detach().cpu().item(),
            -log_likelihood.mean().detach().cpu().item(),
            entropy.detach().cpu().item(),
            value_loss.detach().cpu().item()
        )

class ExDTLoss(LossAbstract):
    def compute_loss(
        self,
        model, 
        states, 
        actions,
        rewards,
        rtg,
        timesteps,
        ordering,
        padding_mask, 
        policy_optimizer,
        value_target,
        value_optimizer,
        inner_epochs,
        clip_range,
        log_temperature_optimizer,
        scheduler=None,
        gamma=0.99
        ):
        state_preds, action_preds, return_preds, value_preds = model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )
        
        # Policy Evaluation
        value_loss = torch.nn.functional.mse_loss(value_preds[padding_mask > 0], value_target.clone()[:, :-1, :][padding_mask > 0]) # Value target starts from 1 to make TD loss
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        action_target = actions.clone()
        
        # Policy Improvement
        # before the inner epoch loop, each action must have the log_prob of the old policy
        # Turn off the dropout layer to stabilize the PPO ratio calculation
        # If there are no eval(), the ratio goes to inf
        model.eval() 
        
        state_preds, action_preds, return_preds, value_preds = model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )
        
        with torch.no_grad():
            behavioral_log_likelihood = action_preds.log_likelihood(action_target)[:, :-1][padding_mask[:, :-1] > 0].detach().clone()

        for inner_epoch in tqdm(range(inner_epochs), desc='PPO loss propagation', position=2, leave=False):
            state_preds, action_preds, return_preds, value_preds = model.forward(
                states,
                actions,
                rewards,
                rtg[:, :-1],
                timesteps,
                ordering,
                padding_mask=padding_mask,
            )
            
            log_likelihood = action_preds.log_likelihood(action_target)[:, :-1][padding_mask[:, :-1] > 0]
            current_step_rewards = rewards[:, :-1, :]
            current_step_values = value_preds.detach().clone()[:, :-1, :] # make it sure to detach the grad graph
            next_step_values = value_preds.detach().clone()[:, 1:, :]
            advantage = current_step_rewards + gamma * next_step_values - current_step_values 
            ratio = torch.exp((log_likelihood - behavioral_log_likelihood).clamp_(max=20)) # preventing nan exploding
            unclipped_loss = -advantage * ratio # adv weighted likelihood without last step
            clipped_loss = -advantage * torch.clamp(
                ratio,
                1. - clip_range,
                1. + clip_range
            )
            entropy = action_preds.entropy().mean()
            ppo_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) # - entropy # 06/05 entropy exploration addition 
            bc_loss = action_preds.log_likelihood(action_target)[padding_mask > 0]
            
            loss = ppo_loss + bc_loss
            policy_optimizer.zero_grad()
            loss.backward() ## TODO is this the optimal solution?
            policy_optimizer.step()
            
        log_temperature_optimizer.zero_grad()
        temperature_loss = (
            model.temperature() * (entropy - model.target_entropy).detach()
        )
        temperature_loss.backward()
        log_temperature_optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        model.train()
        return (
            ppo_loss.detach().cpu().item(),
            -log_likelihood.mean().detach().cpu().item(),
            entropy.detach().cpu().item(),
            value_loss.detach().cpu().item()
        )

def get_loss_function(loss_name: str, pretraining: str) -> LossAbstract:
    if loss_name == 'ODT':
        return ODTLoss(pretraining)
    elif loss_name == 'ODT-Value':
        return ODTValueLoss(pretraining)
    elif loss_name == 'TRPO':
        return TRPOLoss(pretraining)
    elif loss_name == 'PPO':
        return PPOLoss(pretraining)
    elif loss_name == 'ExDT':
        return ExDTLoss(pretraining)
    else:
        raise ValueError(f'Loss function {loss_name} is not supported.')
    

        


