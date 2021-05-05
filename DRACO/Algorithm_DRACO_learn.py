
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
        self.AQ_network = nn.Sequential(
            
            nn.Linear(obs_size, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, n_actions)           
        )
        
        self.sm_layer = nn.Softmax(dim=1)
    
    def forward(self, x):
        aq_network_out = self.AQ_network(x)
        sm_layer_out = self.sm_layer(aq_network_out)
        
        return sm_layer_out


class Exp_Buf():
    
    expbufVar = deque(maxlen=BUF_LEN)


def compute_pherintensity(pheromone_map, X, Y, pa):
    if (X >=1) and (Y>=1):
        pheromone_inpoint = (pheromone_map[X-pa][Y+pa] + pheromone_map[X][Y+pa] + pheromone_map[X+pa][Y+pa]+ \
                         pheromone_map[X-pa][Y]    + pheromone_map[X][Y]    + pheromone_map[X+pa][Y]+ \
                         pheromone_map[X-pa][Y-pa] + pheromone_map[X][Y-pa] + pheromone_map[X+pa][Y-pa])/9
    else: pheromone_inpoint = 0
    
    return pheromone_inpoint
    


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
    eps_decay_steps = 50000 
    
    global_step = 0  
    copy_steps = 100 
    start_steps = 1000 
    steps_train = 4  
   
    batch_size = 32     
    
    n_episodes = 300
   
    gamma = 0.99 
    
    alpha = 0.01 
    
    evap = 0.98
   
    pher_volume = 0.1
   
    pher_intens = 50
   
    pher_area = 1
    
    obs_sizeXY = 4
    ###########################################################################  
  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    qofa_out = n_actions
    
    aq_network_list = []
    tgt_network_list = []
    optimizer_list = []
    objective_list = []
    exp_buf_L = []
    
    for agent_id in range(n_agents):
        
        aq_network = AQ_network(obs_sizeXY, n_actions).to(device)
        
        tgt_network = AQ_network(obs_sizeXY, n_actions).to(device)
       
        exp_buf_L.append(Exp_Buf()) 
        
        aq_network_list.append(aq_network)
        
        tgt_network_list.append(tgt_network)
        
        optimizer_list.append(optim.Adam(params=aq_network_list[agent_id].parameters(), lr=alpha))
        
        objective_list.append(nn.MSELoss())
    
    print ('aq_network_list[0]=', aq_network_list[0])
 
    
    
    ################_for loop by episode _#####################################
    for e in range(n_episodes):
        
        env.reset()
        
        terminated = False
        
        episode_reward = 0
        
        epsilon = max(eps_min, eps_max - (eps_max-eps_min) * global_step/eps_decay_steps)
        print ('epsilon=',epsilon)
       
        pheromone_map = np.zeros([MAPX, MAPY])
         
        map_data = np.zeros((MAPY, MAPX, 3), np.uint8)
        
        
        ######################_while_#####################################
        while not terminated:
                      
            
            actions = []
            action = 0
            
            actionsFox = np.zeros([n_agents]) 
            
            obs_agentXY = np.zeros([n_agents, obs_sizeXY]) 
            obs_agent_nextXY = np.zeros([n_agents, obs_sizeXY])
                        
            
            for agent_id in range(n_agents):
                
                unit = env.get_unit_by_id(agent_id)
                
                pheromone_map[int(unit.pos.x)][int(unit.pos.y)] = pheromone_map[int(unit.pos.x)][int(unit.pos.y)] + pher_volume
                
                cv2.line(map_data, (int(unit.pos.x), int(unit.pos.y)), (int(unit.pos.x), int(unit.pos.y)), (0, 0, pheromone_map[int(unit.pos.x)][int(unit.pos.y)]), 1)
            
            ##############_for agents_#########################################
            for agent_id in range(n_agents):
               
                
                unit = env.get_unit_by_id(agent_id)
                obs_agentXY[agent_id][0] = unit.pos.x
                obs_agentXY[agent_id][1] = unit.pos.y
                
                
                for e_id, e_unit in env.enemies.items():
                    obs_agentXY[agent_id][2] = e_unit.pos.x
                    obs_agentXY[agent_id][3] = e_unit.pos.y
                
                
                obs_agentT = torch.FloatTensor([obs_agentXY[agent_id]]).to(device)
                
                action_probabilitiesT = aq_network_list[agent_id](obs_agentT)
                
                action_probabilitiesT = action_probabilitiesT.to("cpu")
                action_probabilities = action_probabilitiesT.data.numpy()[0]
                 
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                
                action = select_actionFox(action_probabilities, avail_actions_ind, epsilon)
                
                if action is None: action = np.random.choice (avail_actions_ind)
                
                
                if (action == 6):
                   
                    unit = env.get_unit_by_id(agent_id)
                    
                    pheromone_map[int(unit.pos.x)][int(unit.pos.y)] = pheromone_map[int(unit.pos.x)][int(unit.pos.y)] + pher_intens + pher_volume
                    
                                
                actions.append(action)
                actionsFox[agent_id] = action
            ###### end for agents_#############################################

            
            reward, terminated, _ = env.step(actions)
            
            episode_reward += reward
            
            
            flipped = cv2.flip(map_data, 0)
            
            resized = cv2.resize(flipped, dsize=None, fx=10, fy=10)
            
            cv2.imshow('Pheromone map', resized)
            cv2.waitKey(1)
                      
            #####################################
            for agent_id in range(n_agents):
                
                unit = env.get_unit_by_id(agent_id)
                obs_agent_nextXY[agent_id][0] = unit.pos.x
                obs_agent_nextXY[agent_id][1] = unit.pos.y
                
                
                for e_id, e_unit in env.enemies.items():
                    obs_agent_nextXY[agent_id][2] = e_unit.pos.x
                    obs_agent_nextXY[agent_id][3] = e_unit.pos.y
                
                
                pheromone_inpoint = compute_pherintensity (pheromone_map, int(unit.pos.x), int(unit.pos.y), pher_area)
                
                pher_reinf = reward + pheromone_inpoint
                
                exp_buf_L[agent_id].expbufVar.append([obs_agentXY[agent_id], actionsFox[agent_id], obs_agent_nextXY[agent_id], pher_reinf, terminated])
               
                
                ########################_if learning_##########################
                if (global_step % steps_train == 0) and (global_step > start_steps):
                    
                    exp_obs, exp_act, exp_next_obs, exp_pher_reinf, exp_termd = sample_from_expbuf(exp_buf_L[agent_id].expbufVar, batch_size)
                    
                    
                    exp_obs = [x for x in exp_obs]
                    obs_agentT = torch.FloatTensor([exp_obs]).to(device)
                    
                    
                    action_probabilitiesT = aq_network_list[agent_id](obs_agentT)
                    action_probabilitiesT = action_probabilitiesT.to("cpu")
                    action_probabilities = action_probabilitiesT.data.numpy()[0]
                    
                    
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
                    
                    y_batchT = torch.FloatTensor([y_batch64])
                    
                    
                    optimizer_list[agent_id].zero_grad()
                    
                    
                    loss_t = objective_list[agent_id](action_probabilitiesT, y_batchT) 
                    
                                        
                    loss_t.backward()
                    
                    
                    optimizer_list[agent_id].step()
                
                ######################_end if learning_########################
                
                
                if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                    tgt_network_list[agent_id].load_state_dict(aq_network_list[agent_id].state_dict())
            
            ###################################################################
                  
            
            global_step += 1
            
            
            for i in range(MAPX):
                for j in range(MAPY):
                    pheromone_map[i][j] = evap*pheromone_map[i][j] 
           
        ######################_end while_######################################
        
        print('global_step=', global_step, "Total reward in episode {} = {}".format(e, episode_reward))
        
        
        
    ################_end for loop episodes_####################################
    
    
    env.close()
    
    cv2.destroyAllWindows()
    
    
    for agent_id in range(n_agents):
        torch.save(aq_network_list[agent_id].state_dict(),"aqnet_%.0f.dat"%agent_id) 
        
    

 
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s минут ---" % ((time.time() - start_time)/60))