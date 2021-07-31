# Project 2: Continuous Control (Report)

This project trains two agents playing tennis.



### Learning Algorithms:

I used Deep Deterministic Policy Gradients (DDPG) as underlying learning algorithms for training the agent. The implementation is hugely inspired from pendulum implementation provided in udacity deep-reinforcement-learning repo. In essence the program maintains two types of neural networks , actor and critic. The actor network produces action values corresponding to input states , which can help choose best action , while the critic can take that value and produce continuous value for the action taken.What's important  here is  that it takes the good part of both worlds, value based methods and policy gradient methods, and makes them work together in an synchronous way.  The implementation is in line with ddpg algirithm described in the nanodegree course.

Note that in my program , there is only one version of actor and critic networks which are used for both the players. It's quite similar to multi agent continuous control.


### Chosen Hyperparameters:

Below are the hyper parameters which are used in this project:
 <p>

BUFFER_SIZE = int(1e5)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# replay buffer size <br/>
BATCH_SIZE = 256       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# minibatch size<br/>
GAMMA = 0.99            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# discount factor<br/>
TAU = 1e-3              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# for soft update of target parameters<br/>
LR_ACTOR = 1e-4        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# learning rate of the actor <br/>
LR_CRITIC = 1e-4       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# learning rate of the critic<br/>
WEIGHT_DECAY = 0        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# L2 weight decay <br/>
ADAM_EPS = 1e-08        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Adam epsilon (Used for both Networks)
</p>

### Architecture of Neural Network Used:

The actor netowrk has three layers: 
<br/>
— First layer : input size = 33 and output size = 256 <br/>
— Second layer : input size = 256 and output size = 128<br/>
— Third layer : input size = 128 and output size = 4<br/>
<br/>

The critic network has following layers:<br/>
<br/>
— First layer : input size = 33 and output size = 256<br/>
— Second layer : input size = 256 + 4 and output size = 128<br/>
— Third layer : input size = 128 and output size = 1<br/>
<br/>

We use ReLU activation functions for both of the networks.

### Plot of Rewards:

The environment was solved in 100 episodes.

<img src="trainingplot.png"/>

Saved Models: [actormodel.pth](actormodel.pth)
[criticmodel.pth](criticmodel.pth)



### Ideas for Future Work:

Solving the same problem with MADDPG algorithm and comparing it against current implementation.





