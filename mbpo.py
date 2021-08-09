from config import get_config
import os
import random
import argparse

import numpy as np
import torch
import gym
import tqdm
import wandb

import utils
import sac
import transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="MBPO")
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=1e3, type=int)
    parser.add_argument("--max_timesteps", default=400_000, type=int)
    parser.add_argument("--epoch_length", default=1000, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    # SAC
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=int)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--target_entropy", default=-3)
    # Transition
    parser.add_argument("--model_train_freq", default=250, type=int) # TODO:
    parser.add_argument("--model_retain_epochs", default=1, type=int)
    parser.add_argument("--rollout_batch_size", default=100_000, type=int)
    parser.add_argument("--rollout_horizon_schedule", default=None)
    parser.add_argument("--n_train_repeat", default=20, type=int)
    parser.add_argument("--n_random_timesteps", default=5000, type=int)
    parser.add_argument("--real_ratio", default=0.05)

    args = parser.parse_args()

    # update task config
    task_config = get_config(args.env)
    args.max_timesteps = task_config["max_timesteps"]
    args.n_train_repeat = task_config["n_train_repeat"]
    args.rollout_horizon_schedule = task_config["rollout_horizon_schedule"]
    args.target_entropy = task_config["target_entropy"]

    wandb.init(project="mbpo-pytorch")
    wandb.config.update(args)

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    args.seed = random.randint(0, 1000)
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

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
    
    # init model replay buffer
    base_model_buffer_size = int(args.model_retain_epochs
                            * args.rollout_batch_size
                            * args.epoch_length / args.model_train_freq)
    max_model_buffer_size = base_model_buffer_size * args.rollout_horizon_schedule.max_length
    min_model_buffer_size = base_model_buffer_size * args.rollout_horizon_schedule.min_length 
    # init model buffer with the maximal capicity
    model_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, 
                                            max_size=max_model_buffer_size)
    # set the range of model_replay_buffer, this will linearly change during the training
    model_replay_buffer.max_size = min_model_buffer_size

    # init sac
    sac_kwargs = {
        "state_dim": state_dim, 
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau, 
        "target_entropy": args.target_entropy, 
    }
    policy = sac.SAC(**sac_kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")
        print("------ Load policy model ------")

    # init transition
    # termination function
    import static
    task_domain = str(args.env).lower().split("-")[0]
    terminaltion_fn = static[task_domain].termination_fn

    transition_kwargs = {
        "observation_dim": state_dim,
        "action_dim": action_dim,
        "terminal_fn": terminaltion_fn
    }
    transition = transition.Transition(**transition_kwargs)

    if args.load_model != "" :
        transition_file = file_name if args.load_model == "default" else args.load_model
        transition.load(f'./models/{transition_file}')
        print(" ------ Load transition Model ------")

    evaluations = []
    state, done = env.reset(), False
    episode_timesteps = 0

    for t in tqdm.tqdm(range(args.max_timesteps)):
        episode_timesteps += 1

        if t % args.model_train_freq == 0:
            transition_update_info = transition.train(env_replay_buffer)
            
            rollout_horizon = utils.set_rollout_horizon(args.rollout_horizon_schedule, t)
            # change the model buffer size
            model_replay_buffer.max_size = base_model_buffer_size * rollout_horizon
            rollout_info = utils.rollout(rollout_batch_size=args.rollout_batch_size, rollout_horizon=rollout_horizon,
                    transition=transition, policy=policy, env_buffer=env_replay_buffer, model_buffer=model_replay_buffer)

        action = policy.select_action(state, deterministic=False)
        next_state, reward, done, _ = env.step(action)
        done_float = float(done) if episode_timesteps < env._max_episode_steps else 0 # important!
        env_replay_buffer.add(state, action, next_state, reward, done_float)
        
        state = next_state

        if done:
            state, done = env.reset(), False
            episode_timesteps = 0

        if model_replay_buffer.size >= args.batch_size: # have enough data to sample, should always True
            for _ in range(args.n_train_repeat):
                policy_update_info = policy.train(utils.process_sac_data(env_buffer=env_replay_buffer, model_buffer=model_replay_buffer, batch_size=args.batch_size, real_ratio=args.real_ratio))
            wandb.log(policy_update_info)
        
        # Evaluate episode
        if t % args.eval_freq == 0:
            eval_info = utils.eval_policy(policy, args.env, args.seed)
            print(f"Time steps: {t}, Eval_info: {eval_info}")
            wandb.log(eval_info)
            # evaluations.append(utils.eval_policy(policy, args.env, args.seed))
            # np.save(f"./results/{file_name}", evaluations)
            if args.save_model: 
                policy.save(f"./models/{file_name}")
                transition.save(f"./models/{file_name}")