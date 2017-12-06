# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000	

memory = deque(maxlen=2000)
gamma = 0.95    # discount rate
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001

def build_model(input_size, output_size, learning_rate):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=input_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(model, state, output_size):
    if np.random.rand() <= epsilon:
        return random.randrange(output_size)
    act_values = model.predict(state)
    return np.argmax(act_values[0])  # returns action

def replay(model, sample_size):
    global epsilon
    minibatch = random.sample(memory, sample_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + gamma * np.amax(model.predict(next_state)[0]))
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
    	epsilon *= epsilon_decay

    return model

def load(model):
    model.load_weights(model)

def save(model):
    model.save_weights(model)


def main():
    env = gym.make('CartPole-v0')
    env._max_episodes=5000 

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = build_model(input_dim, output_dim, learning_rate)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    sample_size = 64

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, input_dim])
        for time in range(500):
            # env.render()
            action = act(model, state, output_dim)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, input_dim])

            remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, epsilon))
                break

        if len(memory) > sample_size:
            model = replay(model, sample_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

if __name__ == "__main__": main()