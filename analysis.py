import os

import matplotlib.pyplot as plt

best_reward = []
idx = []
path = "Results/ppo_1D_v2"
files = os.listdir(path)
with open(path + "/rewards.txt", 'r') as f:
    for i, r in enumerate(f.read().split(",")):
        if i % 1000 == 0:
            continue

        episode = i + 1
        suffix = '0' * (6-len(str(episode))) + str(episode)
        if os.path.exists(path+"/checkpoint-" + suffix):
            best_reward.append(r)
            idx.append(episode)

plt.plot(idx, best_reward)
plt.show()

