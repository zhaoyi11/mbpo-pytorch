import math
import os
from typing import Any, Optional, Tuple, Sequence

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###### Helper functions ######
def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

def mlp(sizes, ensemble_size, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [EnsembleLinear(sizes[j], sizes[j+1], ensemble_size), act()]
    return nn.Sequential(*layers)


class StandardScaler:
    """ Used to calculate mean, std and normalize data. """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """ Calculate mean and std for given data."""
        self.mean = data.mean(0, keepdim=True) # calculate mean among batch
        self.std = data.std(0, keepdim=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """ Normalization. """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


###### Ensembles ######
class EnsembleLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size

        # register weight and bias. The reason to use it rather than nn.Linear is to expand the ensemble_size dimension
        self.register_parameter('weight', nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))


    def forward(self, x):
        weight = self.weight
        bias = self.bias
        
        # Our linear layer has (ensemble_size, in_features, out_features), there is no strightforward rule to calculate w*x+b. Use torch.einsum to do this.
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias
        return x


class EnsembleMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_features, ensemble_size=7, num_elites=5, with_reward=True):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.with_reward = with_reward
        self.ensemble_size = ensemble_size
        self.num_elites = num_elites
        self.bnn = mlp([self.obs_dim+self.act_dim] + list(hidden_features) + [2 * (self.obs_dim + self.with_reward)],
                        ensemble_size,
                        nn.SiLU)

        self.register_parameter('max_logvar', nn.Parameter(torch.ones( self.obs_dim + self.with_reward) * 0.5, requires_grad=True))
        self.register_parameter('min_logvar', nn.Parameter(torch.ones(self.obs_dim + self.with_reward) * -10, requires_grad=True)) 


    def forward(self,  obs_action):
        """ Given (obs, act), return (next_o, r) """
        _output = self.bnn(obs_action) # return delta_o (and r)
        mu, logvar = torch.chunk(_output, 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar


###### Transition ######
class Transition(object):
    def __init__(self,
                observation_dim:int,
                action_dim:int,
                terminal_fn:Any,
                hidden_dims:Sequence[int]=(200, 200, 200, 200),
                num_ensembles:int=7,
                num_elites:int=5,
                lr:float=3e-4,
                batch_size:int=256,
                max_logging:int=5000,
                early_stop_patience:int=5,
                ):
        super().__init__()

        self.transition = EnsembleMLP(obs_dim=observation_dim, act_dim=action_dim, 
                            hidden_features=hidden_dims, ensemble_size=num_ensembles).to(device)
        self.transition_optim = torch.optim.Adam(self.transition.parameters(), lr=lr, weight_decay=1e-5)
        
        self.num_ensembles = num_ensembles
        self.num_elites = num_elites
        self.lr = lr
        self.batch_size = batch_size
        self.max_logging = max_logging
        self.early_stop_patience = early_stop_patience
        self.terminal_fn = terminal_fn

        self.selected_elites = np.array([i for i in range(num_ensembles)])

        # used to normalized data
        self.scaler = StandardScaler()

    def train(self, replay_buffer:ReplayBuffer, holdout_ratio=0.2):
        """ Train the transition model using the whole replay buffer. """

        def shuffle_rows(arr):
            """ Shuffle among rows. This will keep distinct training for each ensemble."""
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        ############# Process data #############
        data_size = replay_buffer.size
        val_size = min(int(data_size * holdout_ratio), self.max_logging)
        train_size = data_size - val_size

        train_batch, val_batch = replay_buffer.random_split(val_size=val_size)

        train_inputs, train_targets = self._process_training_data(train_batch)
        
        val_inputs, val_targets = self._process_training_data(val_batch)
        val_inputs, val_targets = val_inputs[None].repeat(self.num_ensembles, 1, 1), val_targets[None].repeat(self.num_ensembles, 1, 1)

        # calculate mean and var used for normalizeing inputs
        self.scaler.fit(train_inputs)
        train_inputs, val_inputs = self.scaler.transform(train_inputs), self.scaler.transform(val_inputs) # normalize inputs

        ############# Training Loop #############
        self.val_loss = [1e5 for i in range(self.num_ensembles)]
        epoch, self.cnt, early_stop = 0, 0, False  # cnt is used to count

        idxs = np.random.randint(train_size, size=[self.num_ensembles, train_size])

        while not early_stop:
            for b in range(int(np.ceil(train_size / self.batch_size))):
                batch_idxs = idxs[:, b*self.batch_size : (b+1)*self.batch_size]  # batch_idx.shap: [num_nets, train_batch]
                self.update(train_inputs[batch_idxs, :], train_targets[batch_idxs, :])

            idxs = shuffle_rows(idxs)

            model_loss = self._evaluate(train_inputs[idxs[:, :self.max_logging],:], train_targets[idxs[:, :self.max_logging], :])
            new_val_loss = self._evaluate(val_inputs, val_targets)

            # logging:
            print("In Epoch {e}, the model_loss is : {m}, the val_loss is: {v}".format(e=epoch, m=model_loss, v=new_val_loss))
            early_stop = self._is_early_stop(new_val_loss)
            epoch += 1
        
        ############# End training #############
        val_loss = self._evaluate(val_inputs, val_targets)
        # select model elite according to val error
        sorted_idxs = np.argsort(val_loss)
        self._set_elites(sorted_idxs[:self.num_elites])

        return {'epoch': epoch, 'val_loss_mean': np.mean(val_loss)}

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        """ Perform one step gradient update."""
        mean, logvar = self.transition(inputs)

        inv_var = torch.exp(-logvar)
        mse_loss = (((mean - targets) ** 2) * inv_var).mean(dim=(1,2))
        var_loss = logvar.mean(dim=(1,2))
        loss = (mse_loss + var_loss).sum() # sum over ensembles

        loss += 0.01 * self.transition.max_logvar.sum() - 0.01 * self.transition.min_logvar.sum() 

        self.transition_optim.zero_grad()
        loss.backward()
        self.transition_optim.step()

    def step(self, observations, actions, deterministic:bool=False):
        """ Perform one step forward rollout to predict next observations and rewards.
            Be similar to the openai gym api, the output size is:
                - observation [batch, obs_size]; reward [batch]; done [batch]; info {}

            When inputs are in numpy or non-batched, the ouptouts are also in numpy or non-batched
        """
        return_np = isinstance(observations, np.ndarray)

        assert len(observations.shape) == len(actions.shape)

        if len(observations.shape) == 1:
            observations = observations[None]
            actions = actions[None]
            return_single = True
        else:
            return_single = False

        observations = torch.from_numpy(observations).to(device, torch.float32) if isinstance(observations, np.ndarray) else observations
        actions = torch.from_numpy(actions).to(device, torch.float32) if isinstance(actions, np.ndarray) else actions
        
        obs_act = torch.cat([observations, actions], dim=-1)

        obs_act = self.scaler.transform(obs_act) # normalize inputs
        
        with torch.no_grad():
            ensemble_model_means, ensemble_model_logvars = self.transition(obs_act)  # return the dist (gaussian dist) of next_o and reward
            ensemble_model_stds = torch.sqrt(torch.exp(ensemble_model_logvars))

        if deterministic:
            ensemble_samples = ensemble_model_means
            samples = torch.mean(ensemble_samples, dim=0) # take mean over num_nets

        else:
            ensemble_samples = torch.distributions.Normal(ensemble_model_means, ensemble_model_stds).rsample()
            # random choose one model from ensembles
            num_models, batch_size, _ = ensemble_model_means.shape
            model_idxs = np.random.choice(self.selected_elites, size=batch_size)
            samples = ensemble_samples[model_idxs, np.arange(0, batch_size)]  # different model idx for every data in a batch, shape [batch, n_o+r]

        rewards = samples[:, -1] # reward shape [batch_size]

        next_observations = samples[:, :-1] + observations # next_o = obs_diff + o

        if not self.terminal_fn:
            terminals = torch.zeros_like(rewards).bool()
        else:
            terminals = self.terminal_fn(observations, actions, next_observations).squeeze(dim=-1)
        
        if return_single:  # remove the batch dim
            next_observations = next_observations[0]
            rewards = rewards[0]
            terminals = terminals[0]
        
        if return_np:
            return next_observations.cpu().numpy(), rewards.cpu().numpy(), terminals.cpu().numpy(), {}
        else:
            return next_observations, rewards, terminals, {}

    def save(self, filename):
        torch.save(self.transition.state_dict(), filename + "_transition")
        torch.save(self.transition_optim.state_dict(), filename + "_transition_optimizer")
        torch.save(self.scaler, filename+"_scaler.pth")

    def load(self, filename):   
        self.transition.load_state_dict(torch.load(filename + "_transition"))
        self.transition_optim.load_state_dict(torch.load(filename + "_transition_optimizer"))
        self.scaler = torch.load(filename + "_scaler.pth")

    def _evaluate(self, inputs: torch.Tensor, targets: torch.Tensor)-> np.ndarray:
        with torch.no_grad():
            mean, _ = self.transition(inputs)
            loss = ((mean - targets) ** 2).mean(dim=(1,2))
        return loss.cpu().numpy()

    def _process_training_data(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Pre process training data, change their format to tensor and feed them into device. """
        states, actions, next_states, rewards, not_dones = batch # rewards and not_dones.shape = (batch_size, 1)
        
        inputs = torch.cat([states, actions], dim=-1)
        targets = torch.cat([next_states - states, rewards], dim=-1) # TODO: check reward.shape

        return inputs, targets
    
    def _is_early_stop(self, new_val_loss) -> bool:
        """ To determine whether to early stop or not."""
        changed = False
        for i, old_loss, new_loss in zip(range(len(self.val_loss)), self.val_loss, new_val_loss):
            if (old_loss - new_loss) / old_loss > 0.01:
                changed = True
                self.val_loss[i] = new_loss
        if changed:
            self.cnt = 0
        else:
            self.cnt += 1
        
        if self.cnt >= self.early_stop_patience:
            return True
        else:
            return False    
    
    def _set_elites(self, selected_idxs):
        self.selected_elites = np.array(selected_idxs)

