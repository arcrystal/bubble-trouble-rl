from ray import air, tune
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.tune import TuneConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import (
    PPOTorchRLModule
)
import torch
from game import Game
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def get_module_spec(env, ckpt):
    if ckpt:
        return SingleAgentRLModuleSpec(
            module_class=PPOTorchRLModule,
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_config_dict = {"fcnet_hiddens": [128, 512, 256]},
            catalog_class=PPOCatalog,
            load_state_path=ckpt
        )
    else:
        return SingleAgentRLModuleSpec(
            module_class=PPOTorchRLModule,
            observation_space=env.observation_space,
            action_space=env.action_space,
            catalog_class=PPOCatalog,
            model_config_dict = {"fcnet_hiddens": [128, 512, 256]},
        )
def get_config(env, ckpt=""):
    module_spec = get_module_spec(env, ckpt)
    config = (
        PPOConfig()
        .experimental(_enable_new_api_stack=True)
        .rl_module(
            rl_module_spec=module_spec
        )
        .environment(
            Game,
            env_config={
                "render_mode": None,
                "width": 720
            },
        )
        .framework("torch")
        .training(model={"fcnet_hiddens": [128, 512, 256]})
        .exploration(
            explore=True,
            # exploration_config={
            #     "type": "EpsilonGreedy",
            #     # Parameters for the Exploration class' constructor:
            #     "initial_epsilon": 1.0,
            #     "final_epsilon": 0.02,
            #     "epsilon_timesteps": 1000,  # time-steps over which to anneal epsilon.
            #     "random_timesteps": 100 # time-steps at beginning, over which to act uniformly randomly
            # },
        )
    )
    return config

def tune(config):
    stop = {
        "training_iteration": 1
    }
    tune_config = TuneConfig()
    tune_config.metric = 'episode_reward_mean'
    tune_config.mode = 'max'

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop),
        tune_config=tune_config
    )
    results = tuner.fit()
    return results.get_best_result().config


def train_model(env, episodes=25, print_every=10, ckpt=""):
    name = "ppo"
    version = sum(name in filename for filename in os.listdir("Results/")) + 1
    ppo = get_config(env, ckpt).build()
    save_path = ""
    episode_reward_means = []
    max_reward = -999999
    start = time.time()
    for i in range(1, episodes+1):
        result = ppo.train()
        episode_reward_means.append(result['episode_reward_mean'])
        if i % print_every == 0:
            print(f"\n___________\nEpisode {i}:")
            print(f'Mean reward :', round(result['episode_reward_mean'], 4))
            print(f'Runtime     :', round(time.time() - start, 4) / print_every)
            start = time.time()
        if result['episode_reward_mean'] > max_reward:
            max_reward = result['episode_reward_mean']
            result = ppo.save(os.path.join(os.getcwd(), f"Results/{name}_v{version}"))
            save_path = result.checkpoint.path
            print(f"Ckpt saved  : {save_path}")

    return episode_reward_means, save_path



def plot(rewards):
    name = "ppo"
    version = sum(name in filename for filename in os.listdir("Results/")) + 1
    y = np.array(rewards)
    x = np.arange(len(rewards)) + 1
    plt.plot(x, y)
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.title("Mean Reward per Episode")
    plt.ylabel("Mean Reward")
    plt.xlabel("Episode")
    plt.savefig(f"Plots/ppo_v{version}.png")

def simulate(env, ckpt):
    spec = get_module_spec(env, ckpt)
    module = spec.build()
    action_dist_class = module.get_inference_action_dist_cls()
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    total_steps = 0
    while not terminated:
        fwd_ins = {"obs": torch.Tensor(obs.reshape(1,obs.shape[0]))}
        fwd_outputs = module.forward_exploration(fwd_ins)
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = int(action_dist.sample()[0])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1

    print("Total reward:", total_reward)
    print("Total steps:", total_steps)


if __name__ == "__main__":
    env_config = {'render_mode': None}
    env = Game(env_config)
    rewards, ckpt = train_model(env, episodes=500)
    plot(rewards)
    simulate(env, ckpt=ckpt)
