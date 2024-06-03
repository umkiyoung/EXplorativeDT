from tqdm import trange, tqdm
import torch


def ODT_loss_fn(
    a_hat_dist,
    a,
    attention_mask,
    entropy_reg,
):
    # a_hat is a SquashedNormal Distribution
    log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
    entropy = a_hat_dist.entropy().mean()
    loss = -(log_likelihood + entropy_reg * entropy)
    return (
        loss,
        -log_likelihood,
        entropy,
    )
    
def TRPO_loss_fn(
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

def PPO_loss_fn(
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
    
    info = {}
    
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
        ppo_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        entropy = action_preds.entropy().mean()
        ## TODO which is better? integrated loss or eval - improve iteration?
        
        policy_optimizer.zero_grad()
        ppo_loss.backward() ## TODO is this the optimal solution?
        policy_optimizer.step()
    
    model.train()
    return (
        ppo_loss,
        -log_likelihood.mean(),
        entropy,
        value_loss
    )


def ExDT_loss_fn(
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
    
    info = {}
    
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
        ppo_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        entropy = action_preds.entropy().mean()
        
        bc_loss = action_preds.log_likelihood(action_target)[padding_mask > 0]
        
        ## TODO which is better? integrated loss or eval - improve iteration?
        
        policy_optimizer.zero_grad()
        ppo_loss.backward() ## TODO is this the optimal solution?
        policy_optimizer.step()
        
        
    
    model.train()
    return (
        ppo_loss,
        -log_likelihood.mean(),
        entropy,
        value_loss
    )

losses = {
    'ODT': ODT_loss_fn,
    'TRPO': TRPO_loss_fn,
    'PPO': PPO_loss_fn,
    'ExDT': ExDT_loss_fn
}

