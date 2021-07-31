from unityagents import UnityEnvironment

import numpy as np
import random
import torch
import numpy as np

from collections import deque
import matplotlib.pyplot as plt
import time
from tennisagent import Agent
from tennismemory import ReplayBuffer
   


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ddpg(agent, memory , StopWhenSolved = True):
    
    best_scores = deque(maxlen=100)
    scores_global = []
    average_global = []
    min_global = []
    max_global = []
    time_taken = 0
    Start_time = time.time()
        
    for i_episode in range(1, agent.max_num_episodes):

        env_info = env.reset(train_mode=True)[brain_name]     	# reset the environment    
        states = env_info.vector_observations                  	# get the current state (for each agent)
        scores = np.zeros(num_agents)                          	# initialize the score (for each agent)
        agent.reset()
        
        timestep = 0
        dones = np.zeros(num_agents) 

        while timestep <= agent.T_max:
            timestep += 1 
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished    
						
            agent.memorize(states, actions, rewards, next_states, dones, memory)
			
            agent.learn(memory,timestep)
						
            states = next_states
            scores += rewards                                    # roll over the state to next time step        
			
			             
                    
        episode_avg_score = np.mean(scores)                
        scores_global.append(episode_avg_score)
                
        min_global.append(np.min(scores))  
        max_global.append(np.max(scores)) 
        
        best_scores.append(max_global[len(max_global)-1])        
        best_scores_average = np.mean(best_scores)
        
        
        print('\rEpisode {} \tlast 100 avg: {:.2f} \tavg score: {:.2f} '.format(i_episode, best_scores_average, episode_avg_score), end="")
        if i_episode % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {} \tlast 100 avg: {:.2f}'.format(i_episode, best_scores_average)) 
        
        if  (StopWhenSolved and best_scores_average >= 0.5):            
            End_time = time.time()
            time_taken = (End_time - Start_time)/60
            print('\nSolved in {:d} episodes!\tbest_scores_average: {:.2f}, time taken(min): {}'.
                  format(i_episode, best_scores_average, (End_time - Start_time)/60))
            torch.save(agent.actor_local.state_dict(), 'actormodel.pth')
            torch.save(agent.critic_local.state_dict(), 'criticmodel.pth')            
            break
     
    return scores_global, average_global, max_global, min_global, time_taken



if __name__=='__main__':

    seed = 777 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_path = "C:/projects/deep-reinforcement-learning/p3_collab-compet/Tennis_Windows_x86_64/Tennis.exe"  

    max_num_episodes = 1000    

    memory_capacity = int(1e6)      # Experience replay memory capacity
    batch_size = 256             # Batch size: Number of memories will be sampled to learn 
	
    priority_exponent = 0.5         # Prioritised experience replay exponent (originally denoted α)
    priority_weight = 0.4           # Initial prioritised experience replay importance sampling weight
    
    random.seed(seed)
    torch.manual_seed(random.randint(1, 10000))
    
    torch.cuda.manual_seed(random.randint(1, 10000))

    env = UnityEnvironment(file_name = env_path)
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # Construct the Agent
    agent = Agent(state_size=state_size, action_size=action_size, seed = 777)

    # Replay memory: We create the Replay Mem outside the Agent so we can share 1 single Mem with two agents. 
    memory = ReplayBuffer(action_size, memory_capacity, batch_size, seed)

    scores_global, average_global, max_global, min_global, time_taken = ddpg(agent, memory, max_num_episodes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_global)+1), scores_global)
    plt.plot(np.arange(1, len(average_global)+1), average_global)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(['Episode Avg', 'Last 100 Average'], loc='lower right')
    plt.show()


    env.close()

