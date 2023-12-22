
import ray
from ray import air, tune
from ray.tune import TuneConfig
from ray.rllib.algorithms.ppo import PPOConfig, PPO
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
        "-train", "--train-iters", type=int, default=15, help="Number of iterations to train."
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
                # ValueError: Cannot concat data under key 'obs', b/c sub-structures under that key don't match.
                # `samples`=[SampleBatch(2000 (seqs=5): ['obs', 'new_obs', 'actions', 'prev_actions', 'rewards',
                # 'prev_rewards', 'terminateds', 'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index',
                # 't', 'state_in', 'state_out', 'vf_preds', 'action_dist_inputs', 'action_prob', 'action_logp',
                # 'values_bootstrapped', 'advantages', 'value_targets']),
                # SampleBatch(2000 (seqs=6): ['obs', 'new_obs', 'actions', 'prev_actions', 'rewards',
                # 'prev_rewards', 'terminateds', 'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index',
                # 't', 'state_in', 'state_out', 'vf_preds', 'action_dist_inputs', 'action_prob', 'action_logp',
                # 'values_bootstrapped', 'advantages', 'value_targets'])]

                #  Original error:
                #  all the input array dimensions except for the concatenation axis must match exactly,
                #  but along dimension 1, the array at index 0 has size 512 and the array at index 1 has size 493
                #"max_seq_len": 493, this also failed so trying 256
                "max_seq_len":256,
                #"max_seq_len": tune.grid_search([16, 32, 64, 128, 256]),
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
    config = results.get_best_result().config
    return config


def train_model(train_iters, best_config, print_every=1):
    name = "lstm_ppo"
    version_num = sum(name in file for file in os.listdir("Results/")) + 1
    out_path = f"Results/{name}_v{version_num}"
    algo = PPO(config=best_config)
    start = time.time()
    episode_reward_means = []
    for i in range(train_iters):
        result = algo.train()
        episode_reward_means.append(result['episode_reward_mean'])
        if i % print_every == 0:
            algo.save(out_path)
            print(f"\n___________\nEpisode {i}:")
            print(f'Mean reward:', round(result['episode_reward_mean'], 4))
            print(f'Runtime    : ', round(time.time() - start, 4) / print_every)
            start = time.time()

    with open("rewards.txt", 'w') as f:
        for mean in episode_reward_means:
            f.write(str(mean))
            f.write(',')

        f.close()

    return algo

def plot(filename="rewards.txt"):
    with open(filename, 'r') as f:
        rewards = []
        for x in f.read().split(","):
            if x:
                rewards.append(float(x))

        y = np.array(rewards)
        x = np.array(list(range(len(rewards)))) + 1
        plt.plot(x, y)
        plt.show()


def simulate_model(algo):
    env = Game({'render_mode': 'human', 'fps': 120})
    observation, info = env.reset()
    cumulative_reward = 0
    state = algo.get_state()
    while True:
        action, rnn_state, _ = algo.compute_single_action(observation, state=state)
        print("stepping")
        observation, reward, terminated, truncated, _ = env.step(action)
        state = rnn_state
        if terminated or truncated:
            break

        cumulative_reward += reward

    print(f"Cumulative reward: {cumulative_reward}")

if __name__ == "__main__":
    args = get_args()
    ray.init()
    config = tune_model(args.tune_iters)
    algo = train_model(args.train_iters, config)
    plot()
    simulate_model(algo)
    ray.shutdown()
