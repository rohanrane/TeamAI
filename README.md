# TeamAI
GDMS Reinforcement Learning

We implemented two solution approaches for two diferent OpenAI environments, CartPole and Ms PacMan. At first we set up a basic solution that took a large set of random training data and fed it through our neural network to train it. This approach worked quite well for solving the CartPole environment, however it failed to produce any reasonable results for the PacMan environment. In order to create a method to solve the PacMan environment we used a  Q-Learning implementation of reinforcement learning. This method trains the model with each game, slowly improving over time.
