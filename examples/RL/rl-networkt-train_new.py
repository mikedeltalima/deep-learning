# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%


# %% [markdown]
# # Deep RL network to play pygame "Catcher"
#
# The code below implements the learning algorithm based on the class <code>game_wrapper</code> defined in <code>game_wrapper.py</code>. Run it stand-alone, to see how the game works. In a nutshell, the game has been tuned to move the fruit by one step for every move the paddle makes. Actions are left, still, right, and the rewards at every time step are -1 (lost), 0 (nothing), or +1 (hit paddle).

# %%
# -*- coding: utf-8 -*-
# from https://github.com/PacktPublishing/Deep-Learning-with-Keras/tree/master/Chapter08
from __future__ import division, print_function
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
#from scipy.misc import imresize
from skimage.transform import resize as imresize
import collections
import numpy as np
import os

import wrapped_game
from replay_memory import ReplayMemory, Transition


# initialize parameters
DATA_DIR = "data"
NUM_ACTIONS = 3 # number of valid actions (left, stay, right)
GAMMA = 0.999 # decay rate of past observations
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
MEMORY_SIZE = 50000 # number of previous transitions to remember
NUM_EPOCHS_OBSERVE = 50
NUM_EPOCHS_TRAIN = 1000

BATCH_SIZE = 32
NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN

# %% [markdown]
# The state of the game is given by the last four images. The function below returns s_t, resizing the images to 80x80, normalizing them, and ensuring four images are returned. This is important as the first three steps of the game do not provide a history of four images. For those, the function simply replicates the same image four times.

# %%
def preprocess_images(images):
    if images.shape[0] < 4:
        # single image
        x_t = images[0]
        x_t = imresize(x_t, (80, 80))
        x_t = x_t.astype("float")
        x_t /= 255.0
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    else:
        # 4 images
        xt_list = []
        for i in range(images.shape[0]):
            x_t = imresize(images[i], (80, 80))
            x_t = x_t.astype("float")
            x_t /= 255.0
            xt_list.append(x_t)
        s_t = np.stack((xt_list[0], xt_list[1], xt_list[2], xt_list[3]),
                       axis=2)
    s_t = np.expand_dims(s_t, axis=0)
    return s_t

# %% [markdown]
# This is the core of the experience replay RL algorithm. A batch consists of <code>batch_size</code> state and Q-value pairs (one Q-value per action). These pairs are randomly sampled from all previous experiences.
#
# The state X is a tensor of dimension (batch_size, 80, 80, 4) as every state considers the last four iamges. The training values Y are a tensor of dimension (batch_size, 3), a Q-value for each action.
#
# These Q-values are computed by doing a prediction step using the machine learning model. Note, that two predictions are made: One to compute the Q-values for the current state s_t, the other for the next state s_tp1 that the action a_t has led to. Finally, the algorithm computes the discounted reward based on the actual reward r_t and the discounted reward of the next state. In case the game is done (<code>game_over</code>), there is no future step.

# %%
def get_next_batch(memory, model, num_actions, gamma, batch_size):
    # batch_indices = np.random.randint(low=0, high=len(experience),
    #                                   size=batch_size)
    # batch = [experience[i] for i in batch_indices]

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    X = np.zeros((batch_size, 80, 80, 4))
    Y = np.zeros((batch_size, num_actions))

    s_t, a_t, r_t, s_tp1, game_over = batch
    game_over = np.array(game_over).reshape(batch_size, 1)
    r_t = np.array(r_t)
    X = np.squeeze(np.array(s_t))
    Y = model.predict(X)
    X_tp1 = np.squeeze(np.array(s_tp1))
    Y_tp1 = model.predict(X_tp1)
    Q_sa = np.max(Y_tp1, axis=1)
    def update(idx, y):
        reward = r_t[idx]
        action = a_t[idx]
        if game_over[idx]:
            y[action] = reward
        else:
             y[action] = reward + gamma * Q_sa[idx]
        return y
    Y = np.array([update(i, y) for i, y in enumerate(Y)])
    return X, Y

# %% [markdown]
# The neural network encoding the Q-table is straightforward. A (80,80,4) tensor (the last four images) serves as an input, resulting in three outputs with ReLu activation, resulting into a regression network. No max-pooling layer is used in order to maintain the actual position of the fruit and paddle.

# %%
# build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=8, strides=4,
                 kernel_initializer="normal",
                 padding="same",
                 input_shape=(80, 80, 4)))
model.add(Activation("relu"))
model.add(Conv2D(64, kernel_size=4, strides=2,
                 kernel_initializer="normal",
                 padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, kernel_size=3, strides=1,
                 kernel_initializer="normal",
                 padding="same"))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(512, kernel_initializer="normal"))
model.add(Activation("relu"))
model.add(Dense(3, kernel_initializer="normal"))

model.compile(optimizer=Adam(lr=1e-3), loss="mse")

# %% [markdown]
# Previous games are stored in a Python collection deque of length MEMORY_SIZE. Learning is divided into two steps: an observation phase for the first NUM_EPOCHS_OBSERVE and a training phase.
#
# The key function to run an iteration of the game is <code>game.step(action)</code>. It returns up to four of the last frames in the vector x_t, the reward r_0 and whether the game is over. <code>preprocess_images(x_t)</code> then takes care of padding the images to be at least four as well as normalizing them.
#
# The algorithm calls this function over and over again, selecting a random action for the first NUM_EPOCHS_OBSERVE games, and choosing the best one with probability $1-\epsilon$ otherwise.
#
# In order to find the best action, the neural network performs a prediction step, and the action that is chosen is the one with the highest q-value.
#
# The experience is stored by pushing back the previous state (s_tm1), whatever action has been taken, the reward that resulted, the new state s_t and whether the game is over.
#
# <code>experience.append((s_tm1, a_t, r_t, s_t, game_over))</code>
#
# This experience is then used to incrementally train the neural network model. Finally, the learning algorithm decreases epsilon by a tiny amount.
#

# %%
# train network
game = wrapped_game.MyWrappedGame()
# experience = collections.deque(maxlen=MEMORY_SIZE)
experience = ReplayMemory(MEMORY_SIZE)

num_games, num_wins = 0, 0
epsilon = INITIAL_EPSILON
with open(os.path.join(DATA_DIR, "rl-network-results.tsv"), "w") as fout:
    for e in range(NUM_EPOCHS):
        loss = 0.0
        game.reset()

        # get first state
        a_0 = 1  # (0 = left, 1 = stay, 2 = right)
        x_t, r_0, game_over = game.step(a_0)
        s_t = preprocess_images(x_t)

        while not game_over:
            s_tm1 = s_t
            # next action
            if e <= NUM_EPOCHS_OBSERVE:
                a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
            else:
                if np.random.rand() <= epsilon:
                    a_t = np.random.randint(low=0, high=NUM_ACTIONS, size=1)[0]
                else:
                    q = model.predict(s_t)[0]
                    a_t = np.argmax(q)

            # apply action, get reward
            x_t, r_t, game_over = game.step(a_t)
            s_t = preprocess_images(x_t)
            # if reward, increment num_wins
            if r_t == 1:
                num_wins += 1
            # store experience
            experience.push(s_tm1, a_t, r_t, s_t, game_over)

            if e > NUM_EPOCHS_OBSERVE:
                # finished observing, now start training
                # get next batch
                X, Y = get_next_batch(experience, model, NUM_ACTIONS,
                                    GAMMA, BATCH_SIZE)
                loss += model.train_on_batch(X, Y)

        # reduce epsilon gradually
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPOCHS

        print("Epoch {:04d}/{:d} | Loss {:.5f} | Win Count: {:d}"
            .format(e + 1, NUM_EPOCHS, loss, num_wins))
        fout.write("{:04d}\t{:.5f}\t{:d}\n"
            .format(e + 1, loss, num_wins))

        if e % 100 == 0:
            model.save(os.path.join(DATA_DIR, "rl-network.h5"), overwrite=True)

model.save(os.path.join(DATA_DIR, "rl-network-2100.h5"), overwrite=True)


# %%


