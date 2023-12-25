from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

import torch
import numpy as np
import matplotlib.pyplot as plt

from game import Game

import os
import time
from datetime import datetime, timedelta


def get_module_spec(env, ckpt):
    if ckpt:
        return SingleAgentRLModuleSpec(
            module_class=PPOTorchRLModule,
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_config_dict={"fcnet_hiddens": [128, 512, 256]},
            catalog_class=PPOCatalog,
            load_state_path=ckpt
        )
    else:
        return SingleAgentRLModuleSpec(
            module_class=PPOTorchRLModule,
            observation_space=env.observation_space,
            action_space=env.action_space,
            catalog_class=PPOCatalog,
            model_config_dict={"fcnet_hiddens": [128, 512, 256]},
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
        .training(model={
            "fcnet_hiddens": [128, 512, 256],
            "use_lstm": True,
            "max_seq_len": 64,
            # Size of the LSTM cell.
            "lstm_cell_size": 256,
            # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            "lstm_use_prev_action": True,
            # Whether to feed r_{t-1} to LSTM.
            "lstm_use_prev_reward": True,
        })
        .exploration(
            explore=True,
        )
    )
    return config


# def tune(config):
#     stop = {
#         "training_iteration": 1
#     }
#     tune_config = TuneConfig()
#     tune_config.metric = 'episode_reward_mean'
#     tune_config.mode = 'max'
#
#     tuner = tune.Tuner(
#         "PPO",
#         param_space=config.to_dict(),
#         run_config=air.RunConfig(stop=stop),
#         tune_config=tune_config
#     )
#     results = tuner.fit()
#     return results.get_best_result().config


def train_model(env, episodes=1000, print_every=10, ckpt=""):
    name = "ppo"
    save_path = ""
    episode_reward_means = []
    max_reward = -9999999
    module = get_config(env).build()
    if ckpt:
        module.restore(ckpt)
        start_episode = int(ckpt[-6:]) + 1
        start_idx = ckpt.index('v')+1
        end_idx = ckpt.index('/ckpt')
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

    start_time = time.time()
    for i in range(start_episode, episodes+start_episode):
        result = module.train()
        episode_reward_means.append(result['episode_reward_mean'])
        if i==start_episode: print(result)
        if i % print_every == 0:
            curr_time = time.time() - start_time
            avg_time = curr_time / (i-start_episode+1)
            time_left = avg_time * (episodes+start_episode-i)
            expected_end = datetime.now() + timedelta(seconds=time_left)
            h = int(curr_time // 3600)
            m = int((curr_time - h * 3600) // 60)
            s = round((curr_time - h * 3600 - m * 60))
            print(f"\n___________")
            print(f"Episode {i}/{episodes+start_episode-1}")
            print(f'  Mean reward   :', round(result['episode_reward_mean'], 4))
            print(f'  Mean steps    :', round(result['episode_len_mean'], 4))
            print(f'  Avg runtime   :', round(avg_time, 4))
            print(f'  Total runtime : {h}h {m}m {s}s')
            print(f'  Expected end  : {expected_end.strftime("%H:%M:%S")}')
        if result['episode_reward_mean'] > max_reward:
            if not os.path.exists(path):
                os.mkdir(path)

            max_reward = result['episode_reward_mean']
            prefix = '0' * (6-len(str(i)))
            result = module.save(checkpoint_dir=os.path.join(path, f"ckpt_e{prefix}{i}"))
            save_path = result.checkpoint.path
            print(f"\nCkpt saved    : {save_path}")
            print(f"Mean reward   : {round(max_reward,4)}")

    module.stop()
    with open(path + 'rewards.txt', 'w') as f:
        for value in episode_reward_means:
            f.write(str(value)+',')

    return episode_reward_means, save_path


def plot(rewards):
    version = int(sorted(os.listdir("Results"))[-1][-1])
    y = np.array(rewards)
    x = np.arange(len(rewards)) + 1
    reward_max = max(rewards)
    plt.plot(x, y)
    plt.axhline(y=0.0, color='red', linestyle='-')
    plt.scatter(x=[rewards.index(reward_max)+1],
                y=[max(rewards)],
                color='green')
    plt.text(x=rewards.index(reward_max)+1,
             y=reward_max + 5,
             s=str(round(reward_max, 4)),
             color='green')
    plt.title("Mean Reward per Episode")
    plt.ylabel("Mean Reward")
    plt.xlabel("Episode")
    plt.savefig(f"Results/ppo_v{version}/plot.png")


def simulate(env, ckpt):
    spec = get_module_spec(env, ckpt)
    module = spec.build()
    action_dist_class = module.get_inference_action_dist_cls()
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    total_steps = 0
    while not terminated:
        fwd_ins = {"obs": torch.Tensor(obs.reshape(1, obs.shape[0]))}
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
    checkpoint = ""
    # checkpoint = "Results/ppo_v1"
    # checkpoint += sorted([x for x in os.listdir(ckpt) if "ckpt" in x])[-1]
    game = Game(env_config)
    mean_rewards, checkpoint = train_model(env=game, episodes=10000, print_every=10, ckpt=checkpoint)
    plot(mean_rewards)
    simulate(Game({'render_mode': "human"}), ckpt=checkpoint)
