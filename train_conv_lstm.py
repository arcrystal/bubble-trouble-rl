import os
import json
from pprint import pprint

import ray
from ray import tune
from ray.rllib.algorithms import ppo
from game import Game
import torch

import warnings
warnings.simplefilter("ignore")

def main():
    name = "conv_lstm"
    version_num = sum(name in file for file in os.listdir("Results/"))
    f = open(f"Results/{name}_run{version_num}.json", 'w')

    env_name = "BubbleTrouble"
    tune.register_env(env_name, lambda config: Game())
    ray.init()

    # Create the Algorithm from a config object.
    config = (
        ppo.PPOConfig()
        .framework("torch")
        .environment(env_name)
        .rl_module(_enable_rl_module_api=True)
        .training(
            model={
            #     # "conv_filters": conv_filters,
            #     # "use_lstm": True,
            #     # "max_seq_len": 24,
            #     # "lstm_cell_size": 64,
            #     # "lstm_use_prev_action": True,
            #     # "lstm_use_prev_reward": True,
                 "fcnet_hiddens": [64, 64],
            #     # "fcnet_activation": "tanh",
            #     # "use_attention": False,
            #     # # The number of transformer units within GTrXL.
            #     # # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
            #     # # b) a position-wise MLP.
            #     # "attention_num_transformer_units": 1,
            #     # # The input and output size of each transformer unit.
            #     # "attention_dim": 64,
            #     # # The number of attention heads within the MultiHeadAttention units.
            #     # "attention_num_heads": 1,
            #     # # The dim of a single head (within the MultiHeadAttention units).
            #     # "attention_head_dim": 32,
            #     # # The memory sizes for inference and training.
            #     # "attention_memory_inference": 50,
            #     # "attention_memory_training": 50,
            #     # # The output dim of the position-wise MLP.
            #     # "attention_position_wise_mlp_dim": 32,
            #     # # The initial bias values for the 2 GRU gates within a transformer unit.
            #     # "attention_init_gru_gate_bias": 2.0,
            #     # # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
            #     # "attention_use_n_prev_actions": 0,
            #     # # Whether to feed r_{t-n:t-1} to GTrXL.
            #     # "attention_use_n_prev_rewards": 0,
            },
            _enable_learner_api=True
        )
    )
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("Num GPUs Available:", num_gpus)
        config = config.num_gpus(num_gpus)

    algo = config.build()
    results = {}
    for i in range(10):
        result = algo.train()
        pprint(result)
        for key, value in result.items():
            if key in results:
                results[key].append(value)
            else:
                results[key] = [value]

        if i % 3 == 0:
            checkpoint = algo.save()
            print(f"Checkpoint saved at {checkpoint}")

    algo.stop()
    json.dump(results, f)
    f.close()

if __name__ == "__main__":
    main()
