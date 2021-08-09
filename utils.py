import numpy as np
import torch
import gym
import d4rl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = np.array(state).reshape(1, -1)
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return {'evaluation': avg_reward}

def fill_initial_buffer(env, replay_buffer, n_random_timesteps):
	# pre-fill initial exploration data
	state, done = env.reset(), False
	episode_timesteps = 0
	for _ in range (n_random_timesteps):
		episode_timesteps += 1
		action = env.action_space.sample()
		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0.
		replay_buffer.add(state, action, next_state, reward, done_bool)
		
		state = next_state

		if done:
			state, done = env.reset(), False
			episode_timesteps = 0
			
	return replay_buffer


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
	
	def add_batch(self, states, actions, next_states, rewards, dones):
		for state, action, next_state, reward, done in zip(states, actions, next_states, rewards, dones):
			self.add(state, action, next_state, reward, done)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def get_all(self,):
		return (
			torch.FloatTensor(self.state[:self.size]).to(self.device),
			torch.FloatTensor(self.action[:self.size]).to(self.device),
			torch.FloatTensor(self.next_state[:self.size]).to(self.device),
			torch.FloatTensor(self.reward[:self.size]).to(self.device),
			torch.FloatTensor(self.not_done[:self.size]).to(self.device)
		)
	
	def random_split(self, val_size):
		""" Return training batch and validation batch. Training and validation data are splited randomly."""
		data_size = self.size
		permutation = np.random.permutation(data_size)
		
		training_batch = (torch.FloatTensor(self.state[permutation[val_size:]]).to(self.device),
							torch.FloatTensor(self.action[permutation[val_size:]]).to(self.device),
							torch.FloatTensor(self.next_state[permutation[val_size:]]).to(self.device),
							torch.FloatTensor(self.reward[permutation[val_size:]]).to(self.device),
							torch.FloatTensor(self.not_done[permutation[val_size:]]).to(self.device)
						)		
			
		validation_batch = (torch.FloatTensor(self.state[permutation[:val_size]]).to(self.device),
							torch.FloatTensor(self.action[permutation[:val_size]]).to(self.device),
							torch.FloatTensor(self.next_state[permutation[:val_size]]).to(self.device),
							torch.FloatTensor(self.reward[permutation[:val_size]]).to(self.device),
							torch.FloatTensor(self.not_done[permutation[:val_size]]).to(self.device)
						)						

		return training_batch, validation_batch


########## MBPO uitls ##########

def rollout(rollout_batch_size, rollout_horizon, transition, policy, env_buffer, model_buffer):
    """ Rollout the learned dynamic model to generate imagined transitions. """
    states = env_buffer.sample(rollout_batch_size)[0]
    steps_added = []
    for h in range(rollout_horizon):
        with torch.no_grad():
            action = policy.actor(states, with_logprob=False).to(device)
        next_states, reward, done, info = transition.step(states, action) # shape of r and done: [batch], no additional dim.
        model_buffer.add_batch(states.cpu().numpy(), 
                               action.cpu().numpy(),
                               next_states.cpu().numpy(),
                               reward.cpu().numpy(), 
                               done.cpu().numpy())

        steps_added.append(states.shape[0])          
        nonterm_mask = torch.logical_not(done)
        
        if nonterm_mask.sum() == 0:
            print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(h, nonterm_mask.sum(), nonterm_mask.shape))
            break

        states = next_states[nonterm_mask]

    mean_rollout_length = sum(steps_added) / rollout_batch_size
    return {"Rollout": mean_rollout_length}

def process_sac_data(env_buffer, model_buffer, batch_size, real_ratio):
    """ Cat samples from env_buffer and model_buffer. And suqeeze the r and d (sac update assume no additional dim)
    The returned data format: (o, a, r, d, n_o), and each entry with shape: [env_batch+model_batch, ...]
    """
    env_batch_size = int(batch_size * real_ratio)
    model_batch_size = batch_size - env_batch_size

    env_batch = env_buffer.sample(env_batch_size) # sampled data format: (o, a, r, d, n_o), o.shape: [batch, state_dim], r.shape: [batch, 1]
    model_batch = model_buffer.sample(model_batch_size)
    
    batch = [torch.cat((env_item, model_item), dim=0) for env_item, model_item in zip(env_batch, model_batch)] # cat among batch_size
    return batch

def set_rollout_horizon(rollout_schedule, current_timesteps):
    """ Linearly change the """
    min_length, max_length, min_timesteps, max_timesteps = (rollout_schedule.min_length,
                                                            rollout_schedule.max_length, 
                                                            rollout_schedule.min_timesteps, 
                                                            rollout_schedule.max_timesteps)
    if current_timesteps <= min_timesteps:
        y = min_length
    else:
        dx = (current_timesteps - min_timesteps) / (max_timesteps - min_timesteps)
        dx = min(dx, 1)
        y = dx * (max_length - min_length) + min_length
    return int(y)