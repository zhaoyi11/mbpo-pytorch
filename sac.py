""" Actor Critic Model Implementation."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

device = "cuda" if torch.cuda.is_available() else "cpu"

###### Define Actor and Critic ######
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, act_limit=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.act_limit = act_limit  # The vale limitation of action

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim * 2)
        )

        # constant
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, obs, deterministic=False, with_logprob=True):
        _output = self.actor(obs)
        mean, log_std = torch.chunk(_output, 2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        pi_distribution = distributions.Normal(mean, std)

        if deterministic:
            pi_action_normal = mean
        else:
            pi_action_normal = pi_distribution.rsample()

        pi_action = torch.tanh(pi_action_normal)

        logp_pi = pi_distribution.log_prob(pi_action_normal) - torch.log(1 - pi_action ** 2 + 1e-6)
        logp_pi = logp_pi.sum(1, keepdim=True)

        pi_action = self.act_limit * pi_action

        if with_logprob:
            return pi_action, logp_pi
        else:
            return pi_action

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.input_dim = obs_dim + act_dim
        self.output_dim = 1
        self.hidden_dim = hidden_dim

        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, obs, act):
        inputs = torch.cat((obs, act), dim=-1)
        return self.critic(inputs) # Important! decide whether to squeeze output or not.


class DoubleCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.critic1 = Critic(obs_dim, act_dim, hidden_dim)
        self.critic2 = Critic(obs_dim, act_dim, hidden_dim)
    
    def forward(self, obs, act):
        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        return q1, q2


###### SAC ######
class SAC(object):
    def __init__(
        self, 
        state_dim, 
        action_dim,
        max_action,
        hidden_dim=256,
        discount=0.99,
        tau=0.005, 
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        target_entropy=None,        
    ):
        self.actor = Actor(state_dim, action_dim, hidden_dim=hidden_dim, act_limit=max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = DoubleCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.target_critic = DoubleCritic(state_dim, action_dim, hidden_dim).to(device)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.target_entropy = -action_dim / 2. if target_entropy is None else target_entropy 

        # log_temp, the temperature term for the policy entropy
        self.log_temp = torch.zeros(1, requires_grad=True, device=device)
        self.log_temp_optimizer = torch.optim.Adam([self.log_temp], lr=temp_lr)

        self.discount = discount
        self.tau = tau

        self.total_it = 0

    @property
    def temp(self,):
        return torch.exp(self.log_temp).item()

    def select_action(self, state, deterministic=True):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = state.reshape(1, -1).to(device)

        with torch.no_grad():
            action = self.actor(state, deterministic=deterministic, with_logprob=False)
        return action.cpu().data.numpy().flatten()
    
    def train(self, batch):
        self.total_it += 1

        state, action, next_state, reward, not_done = batch # reward and not_done have size (batch_size, 1)
        ###### fit critic ######
        q1, q2 = self.critic(state, action)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, logp_next_action = self.actor(next_state)
            # Target Q-values
            q1_target, q2_target = self.target_critic(next_state, next_action)
            q_target = torch.min(q1_target, q2_target)
            backup = reward + self.discount * not_done * (q_target - self.temp * logp_next_action)

        # MSE loss against Bellman backup
        critic_loss = ((q1 - backup)**2).mean() + ((q2 - backup)**2).mean()

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ####### fit actor ######
        pi, logp_pi = self.actor(state)
        q1_new, q2_new = self.critic(state, pi)
        q_new = torch.min(q1_new, q2_new)
        # Entropy-regularized policy loss
        actor_loss = (-q_new + self.temp * logp_pi).mean()
        
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
        # ####### fit temp ######
        temp_loss = -torch.mean(self.log_temp * (logp_pi + self.target_entropy).detach())

        # optimize the alpha
        self.log_temp_optimizer.zero_grad()
        temp_loss.backward()
        self.log_temp_optimizer.step()

        ####### update target critic ######
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {"critic_loss": critic_loss.detach().cpu().numpy(),
                "q1": q1.detach().cpu().numpy().mean(),
                "actor_loss": actor_loss.detach().cpu().numpy(),
                "entropy": -logp_pi.detach().cpu().numpy().mean(),
                "temp_loss": temp_loss.detach().cpu().numpy(),
                "temp": self.temp,
                }

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.log_temp, filename+"_log_temp.pth")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.log_temp = torch.load(filename + "_log_temp.pth")


if __name__ == "__main__":
    import argparse
    import gym
    import tqdm
    import wandb

    import utils

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_random_timesteps", default=10_000, type=int)
    parser.add_argument("--max_timesteps", default=1000_000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    args = parser.parse_args()

    wandb.init(project="mbpo-pytorch")
    wandb.config.update(args)

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    env_replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    # prefill random initialization data
    env_replay_buffer = utils.fill_initial_buffer(env, env_replay_buffer, args.n_random_timesteps) # TODO: remove this latter

    # init sac
    sac_kwargs = {
        "state_dim": state_dim, 
        "action_dim": action_dim,
        "max_action": max_action,
    }
    policy = SAC(**sac_kwargs)

    state, done = env.reset(), False
    episode_timesteps = 0

    for t in tqdm.tqdm(range(args.max_timesteps)):
        episode_timesteps += 1

        action = policy.select_action(state, deterministic=False)
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0. # important!
        env_replay_buffer.add(state, action, next_state, reward, done_bool)
        
        state = next_state

        if done:
            state, done = env.reset(), False
            episode_timesteps = 0

        policy_update_info = policy.train(env_replay_buffer.sample(args.batch_size))
        wandb.log(policy_update_info)

        # Evaluate episode
        if t % args.eval_freq == 0:
            eval_info = utils.eval_policy(policy, args.env, args.seed)
            print(f"Time steps: {t}, Eval_info: {eval_info}")
            wandb.log(eval_info)