import os
import gym
import random
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

LR= 1e-3 #learning rate

goal_steps = 500 # number of actions per game
score_req = 50 # want this score at least
initial_games = 10000 # number of games to run

env = gym.make('CartPole-v0').env
env.reset()

def first_game(): #runs random game
    score = 0
    env.reset()
    for i in range(goal_steps):
        env.render() #dont run if you want games to run faster
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score += reward
        if i == 100:
            break

    return score


def initial_pop(): #keeps games that score above score_req
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            #env.render()
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
                
            prev_observation = observation
            score += reward
            if done:
                break
            
        if score >= score_req:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:  
                    output = [1,0]
                training_data.append([data[0], output])
                
        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    
    print 'Average accepted score: ', mean(accepted_scores)
    #print 'Median accepted score: ', median(accepted_scores)
    #print 'Min score : ', min(accepted_scores)
    #print 'Max score : ', max(accepted_scores)
    #print '\n', Counter(accepted_scores)

    counter = Counter([s for s in accepted_scores if s <= 80])
    labels = counter.keys()
    values = counter.values()
    indexes = np.arange(len(labels))
    width = 0.5

    #plt.bar(indexes, values, width)
    #plt.xticks(indexes + width * 0.5, labels)
    #plt.xlabel('Scores')
    #plt.ylabel('Frequency')
    #plt.show() 
    
    return training_data, scores
    

def neural_net_model(input_size): 
    network = input_data(shape = [None, input_size, 1], name = 'input')
    
    network = fully_connected(network, 128, activation = 'relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 128, activation = 'relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 128, activation = 'relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 128, activation = 'relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 128, activation = 'relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 2, activation = 'softmax')
    network = regression(network, optimizer = 'adam', learning_rate=LR, 
                         loss = 'categorical_crossentropy', name = 'targets')
    model = tflearn.DNN(network, tensorboard_dir = 'log')
    
    return model

def train_model(training_data, model = False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]
    
    if not model:
        model = neural_net_model(input_size = len(X[0]))
        
    model.fit({'input':X}, {'targets':y}, n_epoch = 5, snapshot_step = 500,
              show_metric = False, run_id = 'openaistuff')
    
    return model

def run_model(model, iterations):
    scores = []
    choices = []
    for each_game in range(iterations):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            #env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: 
                break

        #print "Game: ", each_game+1, " Score: ", score
        scores.append(score)

    print 'Average Score:', sum(scores)/len(scores)
    #print 'Left: {0:0.4f}   Right: {0:0.4f}'.format(float(choices.count(1))/len(choices), float(choices.count(0))/len(choices))
    return scores

def main():
    training_data, old_scores = initial_pop()
    model = train_model(training_data)
    #model.save("demo.model")
    #model.load("demo.model")
    new_scores = run_model(model, 1000)

    plt.figure()
    plt.hist(old_scores, normed=True, bins=30)
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('')
    plt.show()

    plt.figure()
    plt.hist(new_scores, normed=True, bins=30)
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('')
    plt.show()

    

if __name__ == "__main__": main()
