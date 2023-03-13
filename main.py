from game import Game, DISPLAY_HEIGHT, DISPLAY_WIDTH, FPS
import argparse, os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

# Keras-RL2
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

FRAMES = 30 # num frames for laser to reach ceiling=26
n_features = 128

 # -------------------------- TRAINING RL AGENTS ----------------------
def build_model(n_features, lookback=FRAMES, num_actions=4):
    model = Sequential()
    # 32 Filters of size 8x8
    # stride=how far the filter steps to traverse the img frame
    # model.add(Convolution2D(
    #     32, (8,8), strides=(4,4), activation='relu',
    #     input_shape=(FRAMES, width, height, channels)))
    # model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    # model.add(Convolution2D(64, (3,3), activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(actions, activation='linear'))
    model.add(Flatten(input_shape=(FRAMES, n_features)))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_actions, activation='linear'))
    return model

def build_agent(model, actions, n_warmups):
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=1,
        value_min=0.1, value_test=0.05, nb_steps=1000)

    memory = SequentialMemory(limit=1000000, window_length=FRAMES)
    # dqn = DQNAgent(
    #     model=model, policy=policy, memory=memory,
    #     enable_dueling_network=True, dueling_type='avg',
    #     nb_actions=actions, nb_steps_warmup=1000)
    dqn = DQNAgent(
        model=model, memory=memory, 
        policy=policy, nb_actions=actions, 
        nb_steps_warmup=n_warmups)
    return dqn

def train(visualize, checkpoint=False, n_warmups=1000):
    model = build_model(n_features=n_features, lookback=FRAMES, num_actions=4)
    env = Game(model=model, visualize=visualize, n_features=n_features)
    dqn = build_agent(model, 4, n_warmups)
    dqn.compile(Adam(learning_rate=1e-4))
    if checkpoint:
        dqn.load_weights(checkpoint)
    history = dqn.fit(env, nb_steps=10000, visualize=visualize, verbose=1, action_repetition=1)
    dqn.save_weights("Weights/dqn_laser_lookback")
    return history

def test(checkpoint, episode=1, n_warmups=1000):
    model = build_model(n_features=n_features, lookback=FRAMES, num_actions=4)
    dqn = build_agent(model, 4, n_warmups)
    dqn.compile(Adam(learning_rate=1e-4))
    dqn.load_weights(checkpoint)
    env = Game(n_features=n_features)
    dqn.test(env, nb_episodes=episode)

def main():
    print("------------------------------")
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--User",
        help="Flag to play pygame as user",
        action="store_true")
    parser.add_argument(
        "-tr", "--Train",
        help="Train the DQN Agent to play pygame",
        action="store_true")
    
    parser.add_argument(
        "-te", "--Test",
        help="Test the DQN Agent to play pygame",
        action="store_true")
    parser.add_argument(
        "-m", "--Model",
        help="Start training from model checkpoint")
    args = parser.parse_args()
    # Play game with specified params
    dqn = False
    if args.User:
        game = Game()
        game.play(mode='human')
    if args.Train:
        warmups = 1000
        if args.Model:
            warmups = FRAMES
            ckpt = "Weights/" + args.Model
        else:
            ckpt = False
        history = train(visualize=True, checkpoint=ckpt, n_warmups=warmups)
        print(history)
    if args.Test:
        if args.Model:
            ckpt = "Weights/" + args.Model
            test(ckpt, episode=5)
        else:
            print("Cannot test without model")

if __name__=="__main__":
    main()