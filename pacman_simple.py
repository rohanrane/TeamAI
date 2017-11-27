# install cmake - pip install cmake
# install atari - pip install gym[atari]
            #- pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py

import gym
import numpy as np
from gym.envs.classic_control import rendering
import random
from statistics import mean, median
from collections import Counter

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

viewer = rendering.SimpleImageViewer()
env = gym.make('MsPacman-ram-v0')
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

INIT_EPISODES = 5000 # number of games for pacman to play
score_req = 200
LR = 1e-3 # Learning Rate



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




def first_game(): #runs 1 random game
    env.reset()
    done = False
    score = 0
    while not done:
        large_render() #dont run if you want games to run faster
        action = env.action_space.sample()
        #action = 5
        observation, reward, done, info = env.step(action)
        score += reward
    #print ("Score: ", score)
    #print (len(observation))
    viewer.close()
    
    return score


def initial_pop(): #keeps games that score above score_req
    training_data = []
    scores = []
    accepted_scores = []
    for episode in range(INIT_EPISODES):
        score = 0
        game_memory = []
        prev_observation = []
        done = False
        while not done:
            #env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
                
            prev_observation = observation
            score += reward
            if done:
                break
            
        if score >= score_req:
            accepted_scores.append(score)
            for data in game_memory: #account for each possible action
                if data[1] == 0:
                    output = [1,0,0,0,0,0,0,0,0]
                elif data[1] == 1:
                    output = [0,1,0,0,0,0,0,0,0]
                elif data[1] == 2:
                    output = [0,0,1,0,0,0,0,0,0]
                elif data[1] == 3:
                    output = [0,0,0,1,0,0,0,0,0]
                elif data[1] == 4:
                    output = [0,0,0,0,1,0,0,0,0]
                elif data[1] == 5:
                    output = [0,0,0,0,0,1,0,0,0]
                elif data[1] == 6:
                    output = [0,0,0,0,0,0,1,0,0]
                elif data[1] == 7:
                    output = [0,0,0,0,0,0,0,1,0]
                elif data[1] == 8:
                    output = [0,0,0,0,0,0,0,0,1]
                training_data.append([data[0], output])
                
        print ('\r Episode {}'.format(episode),)
        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('pac_saved.npy', training_data_save)
    
    print ('Average accepted score: ', mean(accepted_scores))
    print ('Median accepted score: ', median(accepted_scores))
    print ('\n', Counter(accepted_scores) )
    
    return training_data



def neural_net_model(input_size): 
    network = input_data(shape = [None, input_size, 1], name = 'input')
    
    network = fully_connected(network, 128, activation = 'relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 256, activation = 'relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 512, activation = 'relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 256, activation = 'relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 128, activation = 'relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 9, activation = 'softmax')
    network = regression(network, optimizer = 'adam', learning_rate=LR, 
                         loss = 'categorical_crossentropy', name = 'targets')
    model = tflearn.DNN(network, tensorboard_dir = 'log')
    
    return model

def train_model(training_data, model = False): #training_data = [prev observation, action array]
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    #print ("X shape, ", X.shape)
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    #print ("y shape, ", np.array(y).shape)
    
    if not model:
        model = neural_net_model(input_size = len(X[0]))
        
    model.fit({'input':X}, {'targets':y}, n_epoch = 3, snapshot_step = 500,
              show_metric = False, run_id = 'openai_pacman')
    
    return model




def main():
    #untrained_score = first_game()
    #print ("\nUntrained Agent Score: ", untrained_score)

    #input("\nPress the <ENTER> key to continue...\n")
    #training_data = initial_pop()
    #viewer.close();

    #input("\nPress the <ENTER> key to continue...\n")
    #model = train_model(initial_pop())
    model = train_model(training_data)
    #model.save("demo.model")
    #model.load("demo.model")

    input("\nPress the <ENTER> key to continue...\n")
    scores = []
    choices = []
    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        done = False
        while not done:
            large_render()

            if len(prev_obs)==0:
                action = random.randrange(0,9)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: 
                break

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

#if __name__ == "__main__": main()












