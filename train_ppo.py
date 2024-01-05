from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
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

def get_module_spec(env, ckpt):
    config_dict = {
        "fcnet_hiddens": fcnet_hiddens,
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


def get_config(env, ckpt=""):
    module_spec = get_module_spec(env, ckpt)
    config = (
        PPOConfig()
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
            rollout_fragment_length=15,
        )
        .environment(
            env.__class__,
            env_config={
                "render_mode": None,
                "width": 720
            }
        )
        .training(
            train_batch_size=150,
            model={
                "fcnet_hiddens": fcnet_hiddens,
                "fcnet_activation": "relu",
            },
            lr=5e-05
        )
    )
    return config


def train_model(env, episodes=1000, print_every=10, save_every=1000, ckpt=""):
    name = f"ppo_{env.name}"
    save_path = ""
    episode_reward_means = []
    max_reward = -100  # minimum reward to trigger saving a checkpoint
    module = get_config(env).build()
    curr_iter_times = np.zeros((print_every,))
    if ckpt:
        module.restore(ckpt)
        start_episode = int(ckpt[-6:]) + 1
        start_idx = ckpt.index('v')+1
        end_idx = ckpt.index('/checkpoint')
        version = int(ckpt[start_idx:end_idx])
        result_path = os.path.join(os.getcwd(), f"Results/{name}_v{version}/")
        with open(result_path + 'rewards.txt', 'r') as f:
            episode_reward_means = [float(x) for x in f.read().split(",") if x]
            max_reward = max(episode_reward_means)
            idx_max = episode_reward_means.index(max_reward)
            episode_reward_means = episode_reward_means[:idx_max+1]
            print("Ckpt highest mean reward:", max_reward)
    else:
        start_episode = 1
        version = sum(name in filename for filename in os.listdir("Results/")) + 1
        result_path = os.path.join(os.getcwd(), f"Results/{name}_v{version}/")

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    for i in range(start_episode, episodes+start_episode):
        result = module.train()
        episode_reward_means.append(result['episode_reward_mean'])
        curr_iter_times[i % print_every] = result['time_this_iter_s']
        with open(result_path+"rewards.txt", 'a') as f:
            f.write(str(result['episode_reward_mean']))
            f.write(",")

        if i == start_episode:
            pprint(result)
        if i % print_every == 0:
            curr_iter_time = curr_iter_times.mean()
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
            print(f'  Mean steps    :', round(result['episode_len_mean'], 4))
            print(f'  Iter runtime  :', round(curr_iter_time, 4))
            print(f'  Total runtime : {d1}d {h1}h {m1}m {s1}s')
            print(f'  Expected end  : {d2}d {h2}h {m2}m {s2}s')
        if result['episode_reward_mean'] > max_reward:
            if not os.path.exists(result_path):
                os.mkdir(result_path)

            max_reward = result['episode_reward_mean']
            prefix = '0' * (6-len(str(i)))
            result = module.save(checkpoint_dir=os.path.join(result_path, f"checkpoint-{prefix}{i}"))
            save_path = result.checkpoint.path
            print(f"\nCkpt saved: {save_path}")
            print(f"Mean reward : {round(max_reward,4)}")
        if i % save_every == 0:
            prefix = '0' * (6-len(str(i)))
            result = module.save(checkpoint_dir=os.path.join(result_path, f"checkpoint-{prefix}{i}"))
            save_path = result.checkpoint.path
            print(f"\nCkpt (not best) saved: {save_path}")

    module.stop()
    with open(result_path + 'rewards.txt', 'w') as f:
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


def simulate(ckpt, n_sims=1):
    env = Game({'render_mode':'human', 'fps':60})
    module = get_config(env).build()
    module.restore(ckpt)
    for n in range(n_sims):
        print("\n__________")
        print("Sim", n)
        obs, info = env.reset()
        terminated = False
        rewards = info
        total_reward = 0
        total_steps = 0
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


fcnet_hiddens = [256, 256, 256]
# fcnet_hiddens = [192, 192, 192]

if __name__ == "__main__":
    best_ckpt = "Results/ppo_1D_v5/checkpoint-007412"
    # ckpt = ""
    # env_config = {
    #     'render_mode': None,
    #     'fps': 24
    # }
    # game = Game(env_config)
    # rewards, best_ckpt = train_model(
    #     env=game,
    #     episodes=25000,
    #     print_every=25,
    #     ckpt=ckpt,
    # )
    # plot(rewards, game)
    simulate(best_ckpt, n_sims=5)
