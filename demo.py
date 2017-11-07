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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

LR= 1e-3 #learning rate

goal_steps = 500 # number of actions per game
score_req = 50 # want this score at least
initial_games = 30000 # number of games to run

env = gym.make('CartPole-v0')
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
    print '\nMedian accepted score: ', median(accepted_scores) 
    #print '\n', Counter(accepted_scores) 
    
    return training_data
    

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
              show_metric = True, run_id = 'openaistuff')
    
    return model

def train_neural_network(x):
    prediction = neural_net_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
    for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)


def main():
    untrained_score = first_game()
    print "\nUntrained Agent Score: ", untrained_score 

    raw_input("\nPress the <ENTER> key to continue...\n")
    training_data = initial_pop()

    raw_input("\nPress the <ENTER> key to continue...\n")
    model = neural_net_model()
    #model = train_model(training_data)
    #model.save("demo.model")
    model.load("demo.model")

    raw_input("\nPress the <ENTER> key to continue...\n")
    scores = []
    choices = []
    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()

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

        scores.append(score)

    print('Average Score:',sum(scores)/len(scores))
    print('Left:{}  Right:{}'.format(float(choices.count(1)/len(choices)), float(choices.count(0)/len(choices))))

if __name__ == "__main__": main()



      
    
    
    
    