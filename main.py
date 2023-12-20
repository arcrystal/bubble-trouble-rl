from game import Game, DISPLAY_HEIGHT, TIMESTEP

import tensorflow as tf
from keras import Model
from keras.models import load_model
from keras.layers import Dense, LSTM, Input, Dropout, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np

import argparse
import math

# num frames for laser to reach ceiling
FRAMES = math.ceil(DISPLAY_HEIGHT / math.floor(DISPLAY_HEIGHT * TIMESTEP)) + 1
n_features = 43

def build_lstm(n_actions):
    inputs = Input(shape=(FRAMES, n_features))
    x = Reshape((FRAMES, -1))(inputs)  # Flatten the input except for the frames dimension
    x = LSTM(512, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(512, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True)(x)
    outputs = Dense(n_actions, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_agent(model, actions, n_warmups):
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps', value_max=1,
        value_min=0.1, value_test=0.05, nb_steps=10000)

    memory = SequentialMemory(limit=1000000, window_length=FRAMES)
    dqn = DQNAgent(
        model=model, policy=policy, memory=memory,
        enable_dueling_network=True, dueling_type='avg',
        nb_actions=actions, nb_steps_warmup=n_warmups)

    # dqn = DQNAgent(
    #     model=model, memory=memory, 
    #     policy=policy, nb_actions=actions, 
    #     nb_steps_warmup=n_warmups, target_model_update=1e-2)
    
    return dqn

def train(visualize, checkpoint, n_steps, n_warmups, outfile, alpha=1e-4):
    model = build_lstm(n_actions=4)
    print(model.summary())
    env = Game(model=model, training=True, n_features=n_features)
    dqn = build_agent(model, 4, n_warmups)
    dqn.compile(Adam(learning_rate=alpha))
    if checkpoint:
        dqn.load_weights(checkpoint)
    dqn.fit(env, nb_steps=n_steps, visualize=visualize, verbose=1, action_repetition=1)
    dqn.save_weights(f"Weights/{outfile}")

def test(checkpoint, episode=1):
    model = build_lstm(n_features=n_features, n_actions=4)
    dqn = build_agent(model, 4, 0)
    dqn.compile(Adam(learning_rate=1e-4))
    dqn.load_weights(checkpoint)
    env = Game(n_features=n_features, training=False)
    dqn.test(env, nb_episodes=episode)

def main():
    import warnings
    warnings.simplefilter('ignore')
    print("------------------------------")
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--User",   action="store_true", help="Flag to play pygame as user")
    parser.add_argument("-tr", "--Train", action="store_true", help="Train the DQN Agent to play pygame")
    parser.add_argument("-te", "--Test",  action="store_true", help="Test the DQN Agent to play pygame")
    parser.add_argument("-m", "--Model",                       help="Start training from model checkpoint")
    parser.add_argument("-s", "--Steps",                       help="Num training steps")
    parser.add_argument("-w", "--Warmup",                      help="Num warmup steps")
    parser.add_argument("-o", "--Outfile",                     help="Checkpoints outfile name")
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
    else:
        from rl.agents import DQNAgent
        from rl.memory import SequentialMemory
        from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
        if args.Train:
            if args.Model:
                ckpt = "Weights/" + args.Model
            else:
                ckpt = False
            if args.Outfile:
                outfile = args.Outfile
            else:
                outfile = 'lstm'
            tf.compat.v1.experimental.output_all_intermediates(True)
            train(visualize=False, checkpoint=ckpt, n_steps=n_steps, n_warmups=n_warmup, outfile=outfile, alpha=1e-4)
        if args.Test:
            if args.Model:
                ckpt = "Weights/" + args.Model
                test(ckpt, episode=50)
            else:
                print("Cannot test without model")


# -------------------------------------------------------


# def main2():
#     tf.compat.v1.experimental.output_all_intermediates(True)
#     frames = FRAMES
#     # Get the input shape and the number of actions
#     input_shape = (frames, n_features)
#     n_actions = 4

#     # Create the Keras model
#     model = build_lstm(n_actions)
#     env = Game(model=model, training=True, n_features=n_features)

#     # Configure and compile the DQN agent
#     memory = SequentialMemory(limit=100000, window_length=frames)
#     policy = EpsGreedyQPolicy(eps=0.1)

#     dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory, nb_steps_warmup=1000, policy=policy, enable_double_dqn=True, dueling_type='avg')
#     dqn.compile(optimizer=Adam())

#     # Train the DQN agent
#     dqn.fit(env, nb_steps=50000, visualize=True, verbose=1)

#     # Save the trained weights
#     dqn.save_weights('dqn_weights.h5f', overwrite=True)

#     # Test the DQN agent
#     dqn.test(env, nb_episodes=5, visualize=True)

def main3():
    play_game = False
    train_model = False
    test_model = True
    if play_game:
        game = Game(frames=FRAMES)
        game.play(mode='human', save_data=True, model=None)

    if train_model:
        X = np.load('X.npy')
        Y = np.load('Y.npy')
        model = build_lstm(n_actions=4)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        callbacks = [ModelCheckpoint(filepath=f'ckpt.h5', save_best_only=True, monitor='loss', mode='min')]
        print(model.summary())
        print(X.shape)
        print(Y.shape)
        model.fit(X, Y, verbose=1, epochs=100, batch_size=128, callbacks=callbacks)
        model.save('lstm.h5')

    if test_model:
        model = load_model('lstm.h5')
        game = Game(frames=FRAMES)
        game.play(mode='human', save_data=False, model=model)
        


if __name__=="__main__":
    main()