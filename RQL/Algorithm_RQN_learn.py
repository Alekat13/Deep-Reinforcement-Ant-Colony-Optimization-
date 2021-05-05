
from smac.env import StarCraft2Env
import numpy as np
from collections import deque

import time

import torch
import torch.nn as nn
import torch.optim as optim

import cv2


np.set_printoptions(threshold=np.inf)


MAPX = 32
MAPY = 32

BUF_LEN = 10000


class AQ_network(nn.Module):
 
    def __init__(self, obs_size, n_actions):
        super(AQ_network, self).__init__()
        self.fc1 = nn.Linear(obs_size, 381)
        self.act1 = nn.ReLU()
        self.fc2 = nn.LSTM(input_size = 381, hidden_size = 70)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(70, n_actions)
       
        self.sm_layer = nn.Softmax(dim=1)

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = x.reshape(-1,1,381)
        x = self.fc2(x)[0]
        x = self.act2(x)
        aq_network_out = self.fc3(x)
        sm_layer_out = self.sm_layer(aq_network_out)
       
        return sm_layer_out


class Exp_Buf():
    
    expbufVar = deque(maxlen=BUF_LEN)



def select_actionFox(action_probabilities, avail_actions_ind, epsilon):
    p = np.random.random(1).squeeze()
    
    if np.random.rand() < epsilon:
        return np.random.choice (avail_actions_ind) 
    else:
        
        for ia in action_probabilities:
            action = np.argmax(action_probabilities)
            if action in avail_actions_ind:
                 return action
            else: 
                action_probabilities[action] = 0
                
      
def sample_from_expbuf(experience_buffer, batch_size):
    
    perm_batch = np.random.permutation(len(experience_buffer))[:batch_size]
   
    experience = np.array(experience_buffer)[perm_batch]
   
    return experience[:,0], experience[:,1], experience[:,2], experience[:,3], experience[:,4]       


def main():

    env = StarCraft2Env(map_name="75z1сFOX", reward_only_positive=False, reward_scale_rate=200, difficulty="1")
    
    env_info = env.get_env_info()
    print ('env_info=',env_info)
  
    obs_size =  env_info.get('obs_shape')
    print ("obs_size=",obs_size)
   
    n_actions = env_info["n_actions"]
    
    n_agents = env_info["n_agents"]
    

    eps_max = 1.0 
    eps_min = 0.1 
    eps_decay_steps = 25000 
    
    global_step = 0 
    copy_steps = 100 
    start_steps = 1000 
    steps_train = 4  
       
    batch_size = 32    
    
    n_episodes = 300
    
    gamma = 0.99 
    
    alpha = 0.01 
    
    
    ###########################################################################  
  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    qofa_out = n_actions
    
    aq_network_list = []
    tgt_network_list = []
    optimizer_list = []
    objective_list = []
    exp_buf_L = []
    
    for agent_id in range(n_agents):
       
        aq_network = AQ_network(obs_size, n_actions).to(device)
       
        tgt_network = AQ_network(obs_size, n_actions).to(device)
        
        exp_buf_L.append(Exp_Buf()) 
       
        aq_network_list.append(aq_network)
        
        tgt_network_list.append(tgt_network)
        
        optimizer_list.append(optim.Adam(params=aq_network_list[agent_id].parameters(), lr=alpha))
        
        objective_list.append(nn.MSELoss())
    
    print ('aq_network_list[0]=', aq_network_list[0])
    
 
    
    
    #####################################################
    for e in range(n_episodes):
        
        env.reset()
       
        terminated = False
        
        episode_reward = 0
        
        epsilon = max(eps_min, eps_max - (eps_max-eps_min) * global_step/eps_decay_steps)
        print ('epsilon=',epsilon)
        
        
        
        
        ###########################################################
        while not terminated:
                      
           
            actions = []
            action = 0
            
            actionsFox = np.zeros([n_agents]) 
            
            obs_agent = np.zeros([n_agents], dtype=object) 
            obs_agent_next = np.zeros([n_agents], dtype=object)
                        
            
            ###################
            for agent_id in range(n_agents):
                
                obs_agent[agent_id] = env.get_obs_agent(agent_id)
                
                obs_agentT = torch.FloatTensor([obs_agent[agent_id]]).to(device)
               
                
                action_probabilitiesT = aq_network_list[agent_id](obs_agentT)
                
                action_probabilitiesT = action_probabilitiesT.to("cpu")
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                 
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                
                action = select_actionFox(action_probabilities, avail_actions_ind, epsilon)
               
                if action is None: action = np.random.choice (avail_actions_ind)
                
               
                              
                actions.append(action)
                actionsFox[agent_id] = action
            ############

            
            reward, terminated, _ = env.step(actions)
           
            episode_reward += reward
            
                                 
            #####################################
            for agent_id in range(n_agents):
                
                obs_agent_next[agent_id] = env.get_obs_agent(agent_id)
                                
                pher_reinf = reward 
               
                exp_buf_L[agent_id].expbufVar.append([obs_agent[agent_id], actionsFox[agent_id], obs_agent_next[agent_id], pher_reinf, terminated])
               
                
                ###########################################
                if (global_step % steps_train == 0) and (global_step > start_steps):
                   
                    exp_obs, exp_act, exp_next_obs, exp_pher_reinf, exp_termd = sample_from_expbuf(exp_buf_L[agent_id].expbufVar, batch_size)
                    
                    
                    exp_obs = [x for x in exp_obs]
                    obs_agentT = torch.FloatTensor([exp_obs]).to(device)
                    
                    
                    action_probabilitiesT = aq_network_list[agent_id](obs_agentT)
                    action_probabilitiesT = action_probabilitiesT.to("cpu")
                    
                    action_probabilities = action_probabilitiesT.data.numpy()#[0]
                    
                     
                    
                    exp_next_obs = [x for x in exp_next_obs]
                    obs_agentT_next = torch.FloatTensor([exp_next_obs]).to(device)
                    
                    
                    action_probabilitiesT_next = tgt_network_list[agent_id](obs_agentT_next)
                    action_probabilitiesT_next = action_probabilitiesT_next.to("cpu")
                    action_probabilities_next = action_probabilitiesT_next.data.numpy()[0]
                    
                    
                    y_batch = exp_pher_reinf + gamma * np.max(action_probabilities_next, axis=-1)*(1 - exp_termd) 
                    
                    
                    y_batch64 = np.zeros([batch_size, qofa_out])
                    
                    for i in range (batch_size):
                        for j in range (qofa_out):
                            y_batch64[i][j] = y_batch[i]
                   
                    
                    y_batchT2 = torch.FloatTensor(y_batch64)
                    
                   
                    a_p32x7T = torch.zeros([batch_size, qofa_out], dtype=torch.float)
                    
                   
                    
                    for i in range (batch_size):
                        for j in range (qofa_out):
                            
                            a_p32x7T[i][j] = action_probabilitiesT[i][0][j]
                    
                   
                    optimizer_list[agent_id].zero_grad()
                    
                  
                    loss_t = objective_list[agent_id](a_p32x7T, y_batchT2) 
                    
                   
                    loss_t.backward()
                    
                    
                    optimizer_list[agent_id].step()
                
                ############################################
                
                
                if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                    tgt_network_list[agent_id].load_state_dict(aq_network_list[agent_id].state_dict())
            
            ##############################
                  
          
            global_step += 1
            
           
           
        ####################################################
        
        print('global_step=', global_step, "Total reward in episode {} = {}".format(e, episode_reward))
        
      
    ###########################################################
    
    
    env.close()
   
    
   
    for agent_id in range(n_agents):
        torch.save(aq_network_list[agent_id].state_dict(),"aqnet_%.0f.dat"%agent_id) 
        
   

if __name__ == "__main__":
    start_time = time.time()
    main()
   
    print("--- %s минут ---" % ((time.time() - start_time)/60))