from collections import namedtuple

Schedule = namedtuple("Schedule", ["min_length", "max_length", "min_timesteps", "max_timesteps"])

def get_config(task):
    env_name = str(task).lower().split('-')[0]
    if env_name == "halfcheetah":
        task_config = {
            "max_timesteps": 400_000,
            "n_train_repeat": 40,
            "rollout_horizon_schedule": Schedule(1, 1, 20_000, 150_000),
            "target_entropy": -3,
        }
    elif env_name == "hopper":
        task_config = {
            "max_timesteps": 125_000,
            "n_train_repeat": 20,
            "rollout_horizon_schedule": Schedule(1, 15, 20_000, 100_00),
            "traget_entropy":-1,
        }
    elif env_name == "walker2d":
        task_config = {
            "max_timesteps": 300_000,
            "n_train_repeat": 20,
            "rollout_horizon_schedule": Schedule(1, 1, 20_000, 150_000),
            "target_entropy": -3,
        }
    elif env_name == "ant":
        task_config = {
            "max_timesteps": 300_000,
            "n_train_repeat": 20,
            "rollout_horizon_schedule": Schedule(1, 25, 20_000, 100_000),
            "target_entropy": -4
        } 
    else:
        raise ValueError
    
    return task_config
