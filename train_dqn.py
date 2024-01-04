from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

from ray.rllib.algorithms.dqn import DQNTorchPolicy
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule



import torch
import numpy as np
import matplotlib.pyplot as plt

from game import Game
from game_lookback import Game2D
from game_lookback_flat import Game2DFlat

import os
from datetime import datetime, timedelta
from pprint import pprint

# Uses the M1 peformance cores instead of the efficiency cores
os.setpriority(os.PRIO_PROCESS, os.getpid(), 1)


def get_module_spec(env, ckpt, model_type="medium"):
    config_dict = {
        "fcnet_hiddens": [300, 300, 300],
        "fcnet_activation": "relu",
    }

    if ckpt:
        return SingleAgentRLModuleSpec(
            module_class=PPOTorchRLModule,
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_config_dict=config_dict,
            catalog_class=PPOCatalog,
            load_state_path=ckpt,
        )
    else:
        return SingleAgentRLModuleSpec(
            module_class=PPOTorchRLModule,
            observation_space=env.observation_space,
            action_space=env.action_space,
            catalog_class=PPOCatalog,
            model_config_dict=config_dict
        )


def get_config(env, ckpt="", model_type='fcnet'):
    module_spec = get_module_spec(env, ckpt, model_type=model_type)
    config = (
        DQNConfig()
        .experimental(
            _enable_new_api_stack=True,
            _disable_preprocessor_api=True
        )
        .rl_module(
            rl_module_spec=module_spec
        )
        .framework("torch")
        .exploration(
            explore=True,
        )
        .rollouts(
            num_rollout_workers=10,
            num_envs_per_worker=1,
            rollout_fragment_length=1000,
        )
        .environment(
            env.__class__,
            env_config={
                "render_mode": None,
                "width": 720
            }
        )
        .training(
            train_batch_size=10000,
            model={
                "fcnet_hiddens": [192, 192, 192],
                "conv_filters": None,
                "fcnet_activation": "relu",
                "use_lstm": model_type=='lstm',
                "max_seq_len": 100,
                "lstm_cell_size": 64,
                # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
                "lstm_use_prev_action": model_type=='lstm',
                # Whether to feed r_{t-1} to LSTM.
                "lstm_use_prev_reward": model_type=='lstm',
            },
            lr=1e-06
        )
    )
    return config


def train_model(env, episodes=1000, print_every=10, ckpt="", model_type="fcnet"):
    name = f"ppo_{env.name}"
    save_path = ""
    episode_reward_means = []
    max_reward = -100  # minimum reward to trigger saving a checkpoint
    module = get_config(env, model_type=model_type).build()
    if ckpt:
        module.restore(ckpt)
        start_episode = int(ckpt[-6:]) + 1
        start_idx = ckpt.index('v')+1
        end_idx = ckpt.index('/checkpoint')
        version = int(ckpt[start_idx:end_idx])
        path = os.path.join(os.getcwd(), f"Results/{name}_v{version}/")
        with open(path + 'rewards.txt', 'r') as f:
            episode_reward_means = [float(x) for x in f.read().split(",") if x]
            max_reward = max(episode_reward_means)
            idx_max = episode_reward_means.index(max_reward)
            episode_reward_means = episode_reward_means[:idx_max+1]
            print("Ckpt highest mean reward:", max_reward)
    else:
        start_episode = 1
        version = sum(name in filename for filename in os.listdir("Results/")) + 1
        path = os.path.join(os.getcwd(), f"Results/{name}_v{version}/")

    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(start_episode, episodes+start_episode):
        result = module.train()
        episode_reward_means.append(result['episode_reward_mean'])
        with open(path+"rewards.txt", 'a') as f:
            f.write(str(result['episode_reward_mean']))
            f.write(",")

        if i == start_episode:
            pprint(result)
        if i % print_every == 0:
            curr_iter_time = result['time_this_iter_s']
            curr_tot_time = result['time_total_s']
            time_left = curr_iter_time * (episodes+start_episode-i)
            d1 = int(curr_tot_time // 86400)
            h1 = int((curr_tot_time - 86400 * d1) // 3600)
            m1 = int((curr_tot_time - 86400 * d1 - 3600 * h1) // 60)
            s1 = round(curr_tot_time - 86400 * d1 - 3600 * h1 - 60 * m1)

            d2 = int(time_left // 86400)
            h2 = int((time_left - 86400 * d2) // 3600)
            m2 = int((time_left - 86400 * d2 - 3600 * h2) // 60)
            s2 = round(time_left - 86400 * d2 - 3600 * h2 - 60 * m2)
            print(f"\n___________")
            print(f"Episode {i}/{episodes+start_episode-1}")
            print(f'  Mean reward   :', round(result['episode_reward_mean'], 4))
            print(f'  All rewards   :', [round(x, 5) for x in result['hist_stats']['episode_reward']])
            print(f'  Mean steps    :', round(result['episode_len_mean'], 4))
            print(f'  Iter runtime  :', round(curr_iter_time, 4))
            print(f'  Total runtime : {d1}d {h1}h {m1}m {s1}s')
            print(f'  Expected end  : {d2}d {h2}h {m2}m {s2}s')
        if result['episode_reward_mean'] > max_reward:
            if not os.path.exists(path):
                os.mkdir(path)

            max_reward = result['episode_reward_mean']
            prefix = '0' * (6-len(str(i)))
            result = module.save(checkpoint_dir=os.path.join(path, f"checkpoint-{prefix}{i}"))
            save_path = result.checkpoint.path
            print(f"\nCkpt saved    : {save_path}")
            print(f"Mean reward     : {round(max_reward,4)}")

    module.stop()
    with open(path + 'rewards.txt', 'w') as f:
        for value in episode_reward_means:
            f.write(str(value)+',')

    return episode_reward_means, save_path


def plot(mean_rewards, env):
    mean_rewards = [x for x in mean_rewards if not np.isnan(x)]
    version = int(sorted(os.listdir("Results"))[-1][-1])
    y = np.array(mean_rewards)
    x = np.arange(len(mean_rewards)) + 1
    max_reward = max(mean_rewards)
    plt.plot(x, y)
    plt.axhline(y=0.0, color='red', linestyle='-')
    plt.scatter(x=[mean_rewards.index(max_reward)+1],
                y=[max(mean_rewards)],
                color='green')
    plt.text(x=mean_rewards.index(max_reward)+1,
             y=max_reward + 5,
             s=str(round(max_reward, 4)),
             color='green')
    plt.title("Mean Reward per Episode")
    plt.ylabel("Mean Reward")
    plt.xlabel("Episode")
    plt.savefig(f"Results/ppo_{env.name}_v{version}/plot.png")


def simulate(ckpt, n_sims=1, model_type="medium"):
    env = Game({'render_mode':'human', 'fps':60})
    module = get_config(env, model_type=model_type).build()
    module.restore(ckpt)
    for n in range(n_sims):
        print("\n__________")
        print("Sim", n)
        obs, info = env.reset()
        terminated = False
        rewards = info
        total_reward = 0
        total_steps = 0
        laser_sim = 0
        nearest_ball = 0
        while not terminated:
            action = module.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            for key, val in info.items():
                rewards[key] += val

            total_reward += reward
            total_steps += 1

        pprint(rewards)
        print("Total steps:", total_steps)
        print("Total reward:", total_reward)


if __name__ == "__main__":
    # env_config = {
    #     'render_mode': None,
    #     'fps': 60
    # }
    # game = Game(env_config)
    # rewards, checkpoint = train_model(
    #     env=game,
    #     episodes=10000,
    #     print_every=10,
    #     ckpt="",
    #     model_type='small'
    # )
    # plot(rewards, game)
    checkpoint = "/Users/acrystal/Desktop/Coding/bubble-trouble-rl/Results/ppo_1D_v7/checkpoint-001091"
    simulate(checkpoint, n_sims=3, model_type="small")
