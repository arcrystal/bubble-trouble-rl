
import gym

class Environment(gym.Env):
    def __init__(self, env_config={}):
        self.observation_space = None # TODO: <gym.space>
        self.action_space = None # TODO: <gym.space> Discrete(3) left, right, shoot

    def reset(self):
        # reset the environment to initial state
        return None # TODO: observation

    def step(self, action):
        # perform one step in the game logic
        return None # TODO: observation, reward, done, info