# q-Learning PacMan


import gym
import numpy as np
from gym.envs.classic_control import rendering
import random
from statistics import mean, median
from collections import Counter, deque
#import matplotlib as mpl
import skimage.transform
import skimage.exposure
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
plt.style.use("ggplot")

import pickle #saves and loads variables

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Flatten, Conv2D
from keras.optimizers import Adam

EPISODES = 5000

viewer = rendering.SimpleImageViewer()
env = gym.make('MsPacman-v0')

memory = deque(maxlen=2000)
gamma = 1    # discount rate
epsilon = 1.5  # exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.0001
sample_size = 50

input_rows = 85
input_columns = 80


output_dim = env.action_space.n

# Possible actions 0-8
# 0: moves left
# 1: moves left then up
# 2: moves right
# 3: moves left
# 4: moves left then down
# 5: moves to 2 o'clock
# 6: moves to 10 o'clock
# 7: moves to 4 o'clock
# 8: moves to 8 o'clock
env.reset()
action=0

#-------------------------------------------------------------------
# Save observation as image file
#from PIL import Image
#im = Image.fromarray(obs2)
#im.save("pacman_gray.png")



#------------------------------------------------------------------
# Env Render Window Handler

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print ("Number of repeats must be larger than 0, k: {k}, l: {l}, returning default array".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

def large_render():
    rgb = env.render('rgb_array')
    upscaled=repeat_upsample(rgb, 3, 3)
    viewer.imshow(upscaled)

def medium_render():
    rgb = env.render('rgb_array')
    upscaled=repeat_upsample(rgb, 2, 2)
    viewer.imshow(upscaled)


# Creates 85 rows by 80 coluns grayscale
def edit_obs2(o):
    o = o[1:171, :]
    o = skimage.color.rgb2gray(o)
    o = skimage.transform.resize(o,(85,80))
    o = skimage.exposure.rescale_intensity(o, out_range=(0, 255))
    return o.astype(int)


#---------------------------------------------------------


def first_game(): #runs 1 random game
    env.reset()
    done = False
    score = 0
    rewards = []
    while not done:
        #large_render() #dont run if you want games to run faster
        action = env.action_space.sample()
        #action = 5
        observation, reward, done, info = env.step(action)
        #observation = edit_obs(observation)
        score += reward
        rewards.append(reward)
    #print ("Score: ", score)
    #obs = observation
    #print (observation)
    viewer.close()
    
    return observation, rewards
obs, r = first_game()



#--------------------------
# New Model
def build_model(input_shape, output_size, learning_rate):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Conv2D(32, 8, strides=(4, 4), data_format="channels_first", activation="relu", input_shape = input_shape))
    model.add(Conv2D(64, 4, strides=(2, 2), data_format="channels_first", activation="relu"))
    model.add(Conv2D(64, 4, strides=(3, 3), data_format="channels_first", activation="relu"))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


#-------------------
#
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(model, state, output_size):
    if np.random.rand() <= epsilon:
        return random.randrange(output_size)
    act_values = model.predict(state)
    return np.argmax(act_values[0])  # returns action

def replay(model, sample_size):
    #print("3")
    global epsilon
    minibatch = random.sample(memory, sample_size)
    for state, action, reward, next_state, done in minibatch:
        #print("4")
        #state = edit_obs(state)
        target = reward
        if not done:
            #print(next_state.shape)
            target = (reward + gamma * np.amax(model.predict(next_state)[0]))
        #print("5")
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
    	epsilon *= epsilon_decay

    return model

def load(model):
    model.load_weights(model)

def save(model, path):
    model.save_weights(model)

#-------------------------------------------------------------------
input_shape = np.reshape(edit_obs2(env.reset()), [1, 85, 80]).shape
#-----------------------------------------------------------
                       
def play_trained_game(numGames, model):
    scores = []
    choices = []
    for each_game in range(numGames):
        score = 0
        #game_memory = []
        first_move = True
        
        state = env.reset()
        state = edit_obs2(state)
        state = np.reshape(state, [1, 1,  input_shape[1], input_shape[2]])
        
        done = False
        while not done:
            #large_render()

            if first_move:
                action = random.randrange(0,9)
                first_move = False
            else:
                action = act(model, state, output_dim)

            next_state, reward, done, _ = env.step(action)
            
            choices.append(action)
            
            next_state = edit_obs2(next_state)
            next_state = np.reshape(next_state, [1, 1, input_shape[1], input_shape[2]])

            score+=reward
            state = next_state
            
        #print (predArr)
        scores.append(score)

    print ('Average Score:', sum(scores)/len(scores))
    print ('0: ', float(choices.count(0))/len(choices),  
        '\n1: ', float(choices.count(1))/len(choices),
        '\n2: ', float(choices.count(2))/len(choices),
        '\n3: ', float(choices.count(3))/len(choices),
        '\n4: ', float(choices.count(4))/len(choices),
        '\n5: ', float(choices.count(5))/len(choices),
        '\n6: ', float(choices.count(6))/len(choices),
        '\n7: ', float(choices.count(7))/len(choices),
        '\n8: ', float(choices.count(8))/len(choices))

    return scores


def graph_histogram(scores, num_bins):
    plt.figure()
    plt.hist(scores, normed=True, bins=num_bins)
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('')
    plt.show()



training_scores = []
def main():    
    
    model = build_model(input_shape, output_dim, learning_rate)
    # agent.load("./save/cartpole-dqn.h5")
    done = False    
    #print("1")
    for e in range(EPISODES):
        state = env.reset()
        state = edit_obs2(state)
        state = np.reshape(state, [1, 1, input_shape[1], input_shape[2]])
        
        score = 0
        done = False
        while not done:
            # env.render()
            action = act(model, state, output_dim)
            next_state, reward, done, _ = env.step(action)
            next_state = edit_obs2(next_state)
            next_state = np.reshape(next_state, [1, 1, input_shape[1], input_shape[2]])

            remember(state, action, reward, next_state, done)

            state = next_state
            score+=reward
            if done:
                break
           
        training_scores.append(score)
        print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, score, epsilon))
        #print("2")
        if len(memory) > sample_size:
            #print("replay call")
            model = replay(model, sample_size)
        # if e % 10 == 0:
        # agent.save("./save/cartpole-dqn.h5")

    plt.plot(training_scores)
    plt.ylabel("Score")
    plt.show

    #model.save('.model')

    input("\nPress the <ENTER> key to continue...\n")
    play_trained_game(5, model)
    
#if __name__ == "__main__": main()


#model = load_model('test.model')

