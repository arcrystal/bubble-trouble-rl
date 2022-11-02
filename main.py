from game import Game, DISPLAY_HEIGHT, DISPLAY_WIDTH, FPS
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

# Keras-RL2
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

FRAMES = int(FPS)

 # -------------------------- TRAINING RL AGENTS ----------------------
def build_model(height, width, channels, actions):
    model = Sequential()
    # 32 Filters of size 8x8
    # stride=how far the filter steps to traverse the img frame
    model.add(Convolution2D(
        32, (8,8), strides=(4,4), activation='relu',
        input_shape=(FRAMES, width, height, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=1,
        value_min=0.1, value_test=0.2, nb_steps=10000)
    # store the past 3 windows for 1000 episodes
    memory = SequentialMemory(limit=1000, window_length=FRAMES)
    dqn = DQNAgent(
        model=model, policy=policy, memory=memory,
        enable_dueling_network=True, dueling_type='avg',
        nb_actions=actions, nb_steps_warmup=10000)
    return dqn

def train():
    env = Game()
    model = build_model(DISPLAY_HEIGHT, DISPLAY_WIDTH, 3, 4)
    dqn = build_agent(model, 4)
    dqn.compile(Adam(learning_rate=1e-4))
    print(model.summary())
    dqn.fit(env, nb_steps=10000, visualize=True, verbose=2)



def main():
    print("------------------------------")
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--User",
        help="Flag to play pygame as user",
        action="store_true")
    parser.add_argument(
        "-t", "--Train",
        help="Train the DQN Agent to play pygame",
        action="store_true")
    args = parser.parse_args()
    
    # Play game with specified params
    if args.User:
        game = Game()
        game.play(mode='human')
    if args.Train:
        train()

if __name__=="__main__":
    main()