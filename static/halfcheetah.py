import torch

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        done = torch.tensor([False]).repeat(len(obs)).to(device=obs.device)
        done = done[:,None]
        return done