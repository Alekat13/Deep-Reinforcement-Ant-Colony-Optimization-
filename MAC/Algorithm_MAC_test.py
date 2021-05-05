
from smac.env import StarCraft2Env
import numpy as np

import torch
import torch.nn as nn


np.set_printoptions(threshold=np.inf)



class MADDPG_Actor(nn.Module):
    
    def __init__(self, obs_size, n_actions):
        super(MADDPG_Actor, self).__init__()
        self.MADDPG_Actor = nn.Sequential(
            
            nn.Linear(obs_size, 381),
            nn.ReLU(),
            nn.Linear(381, 70),
            nn.ReLU(),
            
            nn.Linear(70, n_actions)           
        )
        
        self.sm_layer = nn.Softmax(dim=1)
   
    def forward(self, x):
        aq_network_out = self.MADDPG_Actor(x)
        sm_layer_out = self.sm_layer(aq_network_out)
        
        return sm_layer_out
    

def select_actionFox(action_probabilities, avail_actions_ind, epsilon):
    
    for ia in action_probabilities:
            action = np.argmax(action_probabilities)
            if action in avail_actions_ind:
                return action
            else:
                action_probabilities[action] = 0
 
def main():
    env = StarCraft2Env(map_name="75z1сFOX", difficulty="1")
    env_info = env.get_env_info()
    obs_size =  env_info.get('obs_shape')
    print ("obs_size=",obs_size)
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    
    n_episodes = 50
    epsilon = 0
      
    q_network = MADDPG_Actor(obs_size, n_actions)
    
    q_network_list = []
    
    for agent_id in range(n_agents):
        q_network_list.append(q_network)
        
        state = torch.load("aqnet_%.0f.dat"%agent_id, map_location=lambda stg, _: stg)
        q_network_list[agent_id].load_state_dict(state)
    
    print(q_network_list[0])
    
    
    ############################################################
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        
        
        #############################################################
        while not terminated:
            state = env.get_state()
           
            actions = []
            action = 0
            
            actionsFox = np.zeros([n_agents]) 
            
            obs_agent = np.zeros([n_agents], dtype=object) 
            obs_agent_next = np.zeros([n_agents], dtype=object)
            
            
            for agent_id in range(n_agents):
                ##############################################################
                
                obs_agent[agent_id] = env.get_obs_agent(agent_id)
                
                obs_agentT = torch.FloatTensor([obs_agent[agent_id]])
                  
                action_probabilitiesT = q_network_list[agent_id](obs_agentT)
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
               
                action = select_actionFox(action_probabilities, avail_actions_ind, epsilon)
                if action is None: action = np.random.choice (avail_actions_ind)
                
                actions.append(action)
                actionsFox[agent_id] = action
                ##############################################################

            
            reward, terminated, _ = env.step(actions)
            
            episode_reward += reward
 
        ############################################################
        print("Total reward in episode {} = {}".format(e, episode_reward))
        
       
        
    ###########################################################
    
    
    env.close()
    
       
if __name__ == "__main__":
    main()   