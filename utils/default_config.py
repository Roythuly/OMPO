from utils.config import Config

default_config = Config({
    "seed": 0,
    "tag": "default",
    "start_steps": 5e3,
    "cuda": True,
    "num_steps": 300001,
    "save": True,
    
    "env_name": "HalfCheetah-v2", 
    "eval": True,
    "eval_episodes": 10,
    "eval_times": 10,
    "replay_size": 1000000,
    "local_replay_size": 1000, # default: 1000

    "algo": "TOMAC",
    "policy": "Gaussian",   # 'Policy Type: Gaussian | Deterministic (default: Gaussian)'
    "gamma": 0.99, 
    "tau": 0.005,
    "lr": 0.0003,
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "batch_size": 256, 
    "updates_per_step": 3,
    "target_update_interval": 2,
    "hidden_size": 256,
    "gail_batch": 256,
    
    "exponent": 1.5,  # default: 1.1
    "tomac_alpha": 0.001,  # default: 0.001
    "reward_max": 1.
})
