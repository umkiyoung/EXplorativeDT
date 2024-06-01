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
    a_hat_dist, # action distribution of decision transformer
    a, # action from replay buffer
    values, # for advantage
    attention_mask,
    inner_epochs,
    clip_range
):
    # before the inner epoch loop, each action must have the log_prob of the old policy
    # log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
    behavioral_log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
    
    for inner_epoch in trange(inner_epochs, desc='TRPO loss propagation', position=2, leave=False):
        log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
        advantage = None # need to be implemented
        ratio = torch.exp(log_likelihood - behavioral_log_likelihood)
        unclipped_loss = -advantage * ratio
        clipped_loss = -advantage * torch.clamp(
            ratio,
            1. - clip_range,
            1. + clip_range
        )
        ppo_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        entropy = a_hat_dist.entropy().mean()
        ## TODO if you want to implement PPO logic, optimizer step must be in the inner loop.

        
        return (
            ppo_loss,
            -log_likelihood,
            entropy,
        )


def ExDT_loss_fn(
    a
):
    pass

losses = {
    'ODT': ODT_loss_fn,
    'TRPO': TRPO_loss_fn,
    'PPO': PPO_loss_fn,
    'ExDT': ExDT_loss_fn
}

def TDValueLoss(
    
):
    pass