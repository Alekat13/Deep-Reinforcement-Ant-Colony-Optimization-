
from smac.env import StarCraft2Env
import numpy as np

import torch
import torch.nn as nn

import cv2
from skimage.transform import resize


np.set_printoptions(threshold=np.inf)


MAPX = 32
MAPY = 32


class AQ_network(nn.Module):
    
    def __init__(self, obs_size, n_actions):
        super(AQ_network, self).__init__()
        self.AQ_network = nn.Sequential(
            
            nn.Linear(obs_size, 28),
            nn.ReLU(),
            
            nn.Linear(28, n_actions)           
        )
        
        self.sm_layer = nn.Softmax(dim=1)
    
    def forward(self, x):
        aq_network_out = self.AQ_network(x)
        sm_layer_out = self.sm_layer(aq_network_out)
        
        return sm_layer_out

class CNNnet(nn.Module):   
    def __init__(self):
        super(CNNnet, self).__init__()

        self.cnn_layers = nn.Sequential(
            
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 31)
        )

      
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    

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
    
    obs_sizeXY = 1
  
    q_network = AQ_network(obs_sizeXY, n_actions)
    
    q_network_list = []
    
    
    cnn_network = CNNnet()
    weights_cnn = torch.load("CNNstate3.dat", map_location=lambda stg, _: stg)
    cnn_network.load_state_dict(weights_cnn)
    
    for agent_id in range(n_agents):
        q_network_list.append(q_network)
        
        state = torch.load("aqnet_%.0f.dat"%agent_id, map_location=lambda stg, _: stg)
        q_network_list[agent_id].load_state_dict(state)
    
    print(q_network_list)
    
    
    ##########################################################
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        
        
        ###########################################################
        while not terminated:
            state = env.get_state()
            
            actions = []
            action = 0
            
            actionsFox = np.zeros([n_agents]) 
            
            obs_agentXY = np.zeros([n_agents, obs_sizeXY]) 
            
            
            state_map = np.zeros([MAPX, MAPY])
            
            map_data_state = np.zeros((MAPY, MAPX, 3), np.uint8)
            
            
            actionsFox = np.zeros([n_agents]) 
            
            obs_agentXY = np.zeros([n_agents, obs_sizeXY]) 
            obs_agent_nextXY = np.zeros([n_agents, obs_sizeXY])
            
           
            for e_id, e_unit in env.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                
                cv2.circle(map_data_state, (int(e_x), int(e_y)), 1, (255, 0, 0), -1)
                
           
            for agent_id in range(n_agents):
                
                unit = env.get_unit_by_id(agent_id)
                
                cv2.circle(map_data_state, (int(unit.pos.x), int(unit.pos.y)), 1, (0, 255, 0), -1) 
            
            
            imggrayscale_state = cv2.cvtColor(map_data_state,cv2.COLOR_RGB2GRAY)
            
            img_input = resize(imggrayscale_state, (28,28))
            
            img_input = img_input / 255
            
            
            flipped = cv2.flip(imggrayscale_state, 0)
            
            resized = cv2.resize(flipped, dsize=None, fx=10, fy=10)
            
            cv2.imshow('State map', resized)
            cv2.waitKey(1)
                        
            
            map_data_NNinputR = img_input.reshape(1, 28, 28)
            
            obs_swarmT = torch.FloatTensor([map_data_NNinputR]) #.to(device)
             
            
            stateT = cnn_network(obs_swarmT)
            
            state_probT = torch.exp(stateT) #.cpu()
            
            state_probabilities = state_probT.data.numpy()[0]
            
            state_index = np.argmax(state_probabilities)#, axis=1)
            
            ###################################################################
            
            
            
           
            for agent_id in range(n_agents):
                ##############################################################
                
                obs_agentXY[agent_id][0] = state_index
                
                    
                obs_agentT = torch.FloatTensor([obs_agentXY[agent_id]])
                  
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
 
        ######################цикл while#######################################
        print("Total reward in episode {} = {}".format(e, episode_reward))
        
       
        
    ############################################################
    
   
    env.close()
    
      
if __name__ == "__main__":
    main()   