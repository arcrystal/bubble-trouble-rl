import time

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from game import Game
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)

parser.add_argument(
    "--tune-iters", type=int, default=10, help="Number of iterations to tune."
)
parser.add_argument(
    "--train-iters", type=int, default=1000, help="Number of iterations to train."
)

parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


def tune_model(tune_iters):
    ppo_config = (
        PPOConfig()
        .environment(Game, env_config={"render_mode": None})
        .framework("torch")
        .training(
            model={
                "use_lstm": True,
                "max_seq_len": 64,
                "lstm_use_prev_reward": True,
                "lstm_use_prev_action": True,
            },
            optimizer=(),
            lr=tune.grid_search([0.01, 0.001, 0.0001, 0.00001])
        )
    )

    stop = {
        "training_iteration": tune_iters
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=ppo_config.to_dict(),
        run_config=air.RunConfig(stop=stop),
    )
    results = tuner.fit()
    return results.get_best_result().config


def train_model(train_iters, best_config):
    name = "default"
    version_num = sum(name in file for file in os.listdir("Results/"))
    out_path = f"Results/{name}_v{version_num}"
    print(f"Saving models to {out_path}")
    trainer = PPO(config=best_config)
    start = time.time()
    for i in range(train_iters):
        result = trainer.train()
        if i % 25 == 0:
            trainer.save(out_path)
            print(f"\n___________\nEpisode {i}:")
            print('Episode reward mean     :', round(result['episode_reward_mean'], 4))
            print("25-episode elapsed time :", round(time.time() - start, 4))
            start = time.time()

    return out_path + "/" + sorted(os.listdir(out_path))[-1]


def simulate_model(model_path):
    env_config = {'render_mode': 'human', 'fps': 120}
    model = (PPOConfig()
             .environment(env=Game, env_config=env_config)
             .build())
    model.restore(model_path)
    env = Game(env_config)
    observation, info = env.reset()
    cumulative_reward = 0
    while True:
        action = model.compute_single_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

        cumulative_reward += reward

    print(f"Cumulative reward: {cumulative_reward}")


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    ray.init(local_mode=False)
    config = tune_model(args.tune_iters)
    path = train_model(args.train_iters, config)
    simulate_model(path)
    ray.shutdown()
