# Project 1 Report

## Learning Algorithm

In this project, two agents are trained Agent and AgentP, representing the [Double Deep Q-Netowork](https://arxiv.org/abs/1509.06461) and double DQN with [Prioritized experience replay](https://arxiv.org/abs/1511.05952) respectively.

### `dqn_agent.py`
This file define the agent using the [double DQN](https://arxiv.org/abs/1509.06461). It utilises Double Q-learning to reduce overestimation by decomposing the max operation in the target into action selection and action evaluation. Its update is the same as for DQN, but replacing the target with: ![Screenshot from 2021-05-24 14-56-12](https://user-images.githubusercontent.com/35868876/119309099-6b8b1b80-bca0-11eb-85ae-b7a7de7fc868.png)

We are still estimating the value of the greedy policy according to the current weights. However, evalutated using another set of parameters. The target parameters will be updated by the local parameters using soft update (`TAU = 1e-3 `).

The minibatch size is 32, so the model will learn once the ReplayBuffer contains 32 samples. It will learn every 4 samples are added into the buffer `UPDATE_EVERY = 4`.
The discount factor is 0.99. The Adam optimizer is used with a learning rate 0.0005.

### `model.py`
The input of the network is 37 as the state size is 37. The first hidden layer has  64 nodes, while the second hidden layer also contain 64 nodes. The output layer is 4 which is the action size. All these layers are separated by Rectifier Linear Units (ReLu).
