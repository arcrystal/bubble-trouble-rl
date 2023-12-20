import ray
from ray import tune
from ray.rllib.algorithms import ppo
from ray.tune.logger import pretty_print
from game import Game
import warnings
warnings.simplefilter("ignore")

if __name__ == "__main__":
    f = open("out.txt", 'w')
    env_name = "BubbleTrouble"
    tune.register_env(env_name, lambda config: Game())
    conv_filters = [
        [16, [8, 8], 4],  # 16 filters, 8x8 kernel, stride 4
        [32, [4, 4], 2],  # 32 filters, 4x4 kernel, stride 2
        [64, [3, 3], 2],  # 64 filters, 3x3 kernel, stride 2
        # Add more layers if necessary
    ]

    ray.init()

    # Create the Algorithm from a config object.
    config = (
        ppo.PPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .training(
            model={
                "conv_filters": conv_filters,
                "use_lstm": True,
                "max_seq_len": 48,
                "lstm_cell_size": 256,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": True,
            },
            _enable_learner_api=True,
        )
    )
    algo = config.build()
    for i in range(300):
        result = algo.train()
        f.write(f"Iteration {i}:\n{pretty_print(result)}")
        f.write("-----------------------\n----------------------\n\n")
        print(f"Iteration {i} reward: {result['episode_reward_mean']}")

        if i % 25 == 0:
            checkpoint = algo.save()
            print(f"Checkpoint saved at {checkpoint}")

    algo.evaluate()
    algo.stop()
    f.close()
