from game import Game, DISPLAY_HEIGHT, TIMESTEP

from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, Conv2D, LSTM, Input
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

import argparse
import math

# num frames for laser to reach ceiling
FRAMES = math.ceil(DISPLAY_HEIGHT / math.floor(DISPLAY_HEIGHT * TIMESTEP)) + 1
n_features = 50

def build_model(n_features, n_actions=4):
    model = Sequential()
    model.add(Flatten(input_shape=(FRAMES, n_features)))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(n_actions, activation='linear'))
    return model

def build_drqn(n_actions):
    in_frame = Input(shape=(None, 42, 84, 1))

    conv1 = TimeDistributed(Conv2D(32, (8, 8), strides=4, activation='relu'))(in_frame)
    conv2 = TimeDistributed(Conv2D(64, (4, 4), strides=2, activation='relu'))(conv1)
    conv3 = TimeDistributed(Conv2D(64, (3, 3), strides=1, activation='relu'))(conv2)

    flatten = TimeDistributed(Flatten())(conv3)
    lstm1 = LSTM(256, return_sequences=True)(flatten)
    lstm2 = LSTM(256, return_sequences=False)(lstm1)
    dense = Dense(n_actions, activation='softmax')(lstm2)
    
    model = Model(inputs=in_frame, outputs=dense)
    model.compile(loss="categorical_crossentropy",
                  optimizer='adam',
                  metrics=["accuracy"]
    )
    return model

def build_agent(model, actions, n_warmups):
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=1,
        value_min=0.1, value_test=0.05, nb_steps=10000)

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

def train(visualize, checkpoint, n_steps, n_warmups, outfile, alpha=1e-4):
    model = build_drqn(n_actions=4)
    print(model.summary())
    env = Game(model=model, training=True, n_features=n_features)
    dqn = build_agent(model, 4, n_warmups)
    dqn.compile(Adam(learning_rate=alpha))
    if checkpoint:
        dqn.load_weights(checkpoint)
    dqn.fit(env, nb_steps=n_steps, visualize=visualize, verbose=1, action_repetition=1)
    dqn.save_weights(f"Weights/{outfile}")

def test(checkpoint, episode=1):
    model = build_model(n_features=n_features, n_actions=4)
    dqn = build_agent(model, 4, 0)
    dqn.compile(Adam(learning_rate=1e-4))
    dqn.load_weights(checkpoint)
    env = Game(n_features=n_features, training=False)
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
    parser.add_argument(
        "-s", "--Steps",
        help="Num training steps")
    parser.add_argument(
        "-w", "--Warmup",
        help="Num warmup steps")
    parser.add_argument(
        "-o", "--Outfile",
        help="Checkpoints outfile name")
    args = parser.parse_args()
    try:
        n_steps = int(args.Steps)
    except TypeError:
        n_steps = 10000
    try:
        n_warmup = int(args.Warmup)
    except TypeError:
        n_warmup = 1000
        
    # Play game with specified params
    if args.User:
        game = Game()
        game.play(mode='human')
    if args.Train:
        if args.Model:
            ckpt = "Weights/" + args.Model
        else:
            ckpt = False
        if args.Outfile:
            outfile = args.Outfile
        else:
            outfile = 'dqn_laser_lookback'
        train(visualize=True, checkpoint=ckpt, n_steps=n_steps, n_warmups=n_warmup, outfile=outfile, alpha=0.01)
    if args.Test:
        if args.Model:
            ckpt = "Weights/" + args.Model
            test(ckpt, episode=50)
        else:
            print("Cannot test without model")

if __name__=="__main__":
    main()