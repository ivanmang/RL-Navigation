# Project 1 Report

## Learning Algorithm

In this project, two agents are trained Agent and AgentP, representing the [Double Deep Q-Netowork](https://arxiv.org/abs/1509.06461) and double DQN with [Prioritized experience replay](https://arxiv.org/abs/1511.05952) respectively.

### `dqn_agent.py`
This file defines the agent using the [double DQN](https://arxiv.org/abs/1509.06461). It utilises Double Q-learning to reduce overestimation by decomposing the max 
operation in the target into action selection and action evaluation. Its update is the same as for DQN, but replacing the target with: ![Screenshot from 2021-05-24 14-56-12](https://user-images.githubusercontent.com/35868876/119309099-6b8b1b80-bca0-11eb-85ae-b7a7de7fc868.png)

We are still estimating the value of the greedy policy according to the current weights. However, evalutated using another set of parameters. The target parameters will be updated by the local parameters using soft update (`TAU = 1e-3 `).

The minibatch size is 32, so the model will learn once the ReplayBuffer contains 32 samples. It will learn every 4 samples are added into the buffer `UPDATE_EVERY = 4`.
The discount factor is 0.99. The Adam optimizer is used with a learning rate 0.0005.

### `model.py`
The input of the network is 37 as the state size is 37. The first hidden layer has  64 nodes, while the second hidden layer also contain 64 nodes. The output layer is 4 which is the action size. All these layers are separated by Rectifier Linear Units (ReLu). Both of the agents use this model.

### `dqn_agent_pp.py`
This file defines a agent using [Double DQN](https://arxiv.org/abs/1509.06461) together with [Prioritized experience replay](https://arxiv.org/abs/1511.05952). It used prioritized sampling, that weigh samples so that the samples with a high td-error will be drawn more frequetnly for training. We define `ALPHA` to control how much we want priority and randomness in sampling.  However, we also need to modify our update rule on weight since we are using a non-uniform sampling and bias in Q-value will be produced. Therefore, we need to implement importance sampling weight and we define `BETA` to control how much the sampling weight affect learning. The hyperparameters of this agent is similar to the `dqn_agent.py`, except we added `ALPHA = 0.3` and `BETA = 0.4` for the PER control



<img width="500" alt="Screen_Shot_2020-06-03_at_2 37 00_PM_30SiARt" src="https://user-images.githubusercontent.com/35868876/119313289-f4f11c80-bca5-11eb-9738-a54bf7d28fce.png">

### Result
The agent using the double DQN only able to solve the environment in 533 episodes.
`Environment solved in 533 episodes!	Average Score: 13.00`

while the agent using Double DQN together with Prioritized experience replay solve it in 662 episodes.
`Environment solved in 662 episodes!	Average Score: 13.01`

![Screenshot from 2021-05-24 20-14-00](https://user-images.githubusercontent.com/35868876/119346291-9c814580-bccc-11eb-8eb9-f65d20491c20.png)


The blue line represent agent using the double DQN, while the other is represented in orange. We can see that the first agent have a better performance.


### Future work
We believe some more works can be done to improve agents performance:
1. Implement a dueling DQN
2. Use [SumTree](https://github.com/rlcode/per/blob/master/SumTree.py) to store the priority of the sample
3. Redesign the neural network in `model.py`
