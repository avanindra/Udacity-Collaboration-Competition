import numpy as np
import random
import copy

from tennisnetwork import Actor, Critic


import torch    
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4        # learning rate of the actor 
LR_CRITIC = 1e-4       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
REWARD_CLIP = 1
ADAM_EPS = 1e-08                # Adam epsilon (Used for both Networks)


memory_capacity = int(1e6)      # Experience replay memory capacity
learning_starts_ratio = 1/50   # Number of steps before starting training = memory capacity * this ratio
learning_frequency = 2    # Steps before we sample from the replay Memory again 




class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size , seed ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            
        """
        # self.seed = seed
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.T_max = 1000
        self.max_num_episodes = int(1000)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR , eps=ADAM_EPS)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY) 

        # Noise process
        self.noise = OUNoise(action_size, seed, sigma=0.05)

        self.learning_starts =  int(memory_capacity * learning_starts_ratio) 
        self.learning_frequency = learning_frequency       
    
    def memorize(self, states, actions, rewards, next_states, dones, memory):
        """Save experience in replay memory, and use random sample from buffer to learn."""        
        # Save experience / reward
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):        
            memory.add(state, action, reward, next_state, done)

       
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, memory, timestep):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            
        """
        # Learn, if enough samples are available in memory and after "learning_frequency" steps since we last learnt                    
        if len(memory) > self.learning_starts and timestep % self.learning_frequency == 0:           
            states, actions, rewards, next_states, dones = memory.sample()
                        
            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_local(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)        
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()        
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), REWARD_CLIP)
            self.critic_optimizer.step()
            

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            
            # ----------------------- update target networks ----------------------- #
            # Every time there is a leartning process happening, let's update
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)                     


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
