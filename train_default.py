import ray
from ray import air, tune
from ray.tune import TuneConfig
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.algorithm import Algorithm
from game import Game
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )

    parser.add_argument(
        "-tune", "--tune-iters", type=int, default=1, help="Number of iterations to tune."
    )
    parser.add_argument(
        "-train", "--train-iters", type=int, default=1, help="Number of iterations to train."
    )

    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )
    return parser.parse_args()


def tune_model(tune_iters):
    ppo_config = (
        PPOConfig()
        .environment(Game, env_config={"render_mode": None})
        .framework("torch")
        .training(
            model={
                "use_lstm": True,
                "max_seq_len": 256,
                # "max_seq_len": tune.grid_search([16, 32, 64, 128, 256]),
                "lstm_use_prev_reward": True,
                "lstm_use_prev_action": True,
            },
            # lr=tune.grid_search([0.1,     0.05,
            #                      0.01,    0.005,
            #                      0.001,   0.0005,
            #                      0.0001,  0.00005,
            #                      0.00001, 0.000005])
            lr=0.005
        )
    )

    stop = {
        "training_iteration": tune_iters
    }
    tune_config = TuneConfig()
    tune_config.metric = 'episode_reward_mean'
    tune_config.mode = 'max'

    tuner = tune.Tuner(
        "PPO",
        param_space=ppo_config.to_dict(),
        run_config=air.RunConfig(stop=stop),
        tune_config=tune_config
    )
    results = tuner.fit()
    return results.get_best_result().config


def train_model(train_iters, best_config, print_every=1, ckpt=""):
    name = "lstm_ppo"
    version_num = sum(name in file for file in os.listdir("Results/")) + 1
    out_path = f"Results/{name}_v{version_num}"
    ppo = PPO(config=best_config)
    if ckpt:
        ppo.restore(ckpt)

    start = time.time()
    episode_reward_means = []
    for i in range(train_iters):
        result = ppo.train()
        episode_reward_means.append(result['episode_reward_mean'])
        if i % print_every == 0:
            checkpoint_dir = ppo.save().checkpoint.path
            print(f"\n___________\nEpisode {i}:")
            print(f"Ckpt saved at {checkpoint_dir}")
            print(f'Mean reward:', round(result['episode_reward_mean'], 4))
            print(f'Runtime    : ', round(time.time() - start, 4) / print_every)
            start = time.time()

    with open("rewards.txt", 'w') as f:
        for mean in episode_reward_means:
            f.write(str(mean))
            f.write(',')

        f.close()

    return ppo


def plot(filename="rewards.txt"):
    with open(filename, 'r') as f:
        rewards = []
        for x in f.read().split(","):
            if x:
                rewards.append(float(x))

        y = np.array(rewards)
        x = np.array(list(range(len(rewards)))) + 1
        plt.plot(x, y)
        plt.axhline(y=0.0, color='r', linestyle='-')
        plt.title("Mean Reward per Episode")
        plt.ylabel("Mean Reward")
        plt.xlabel("Episode")
        name = "lstm_ppo"
        version_num = sum(name in file for file in os.listdir("Results/")) + 1
        plt.savefig(f"Plots/{name}_v{version_num}.png")


def simulate_model(ckpt):
    algo = Algorithm.from_checkpoint(ckpt)
    policy = algo.get_policy()
    env = Game({'render_mode': 'human', 'fps': 120})
    observation, info = env.reset()
    cumulative_reward = 0
    state = policy.get_initial_state()
    while True:
        action, rnn_state, _ = policy.compute_single_action(observation, state=state)
        observation, reward, terminated, truncated, _ = env.step(action)
        state = rnn_state
        if terminated or truncated:
            break

        cumulative_reward += reward
        print(f"Cumulative reward: {cumulative_reward}")


if __name__ == "__main__":
    project_dir = "/Users/acrystal/Desktop/Coding/bubble-trouble-rl/"
    ckpt = project_dir + "Results/lstm_ppo_v9/checkpoint_000025"
    args = get_args()
    ray.init()
    config = tune_model(args.tune_iters)
    ppo_algo = train_model(args.train_iters, config, ckpt=ckpt)
    plot()
    simulate_model(ckpt)
    ray.shutdown()
