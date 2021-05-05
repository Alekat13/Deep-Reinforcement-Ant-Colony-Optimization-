
from smac.env import StarCraft2Env
import numpy as np
from collections import deque

import time

import torch
import torch.nn as nn
import torch.optim as optim


np.set_printoptions(threshold=np.inf)


MAPX = 32
MAPY = 32

BUF_LEN = 10000


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


class MADDPG_Critic(nn.Module):
    def __init__(self, full_obs_size, n_actions_agents):
        super(MADDPG_Critic, self).__init__()
       
        self.network = nn.Sequential(
             
            nn.Linear(full_obs_size+n_actions_agents, 2000),
            nn.ReLU(),
            
            nn.Linear(2000, 1000),
            nn.ReLU(),
           
            nn.Linear(1000, 100),
            nn.ReLU(),
           
            nn.Linear(100, 1)
            )
    
    def forward(self, state, action):
       
        x = torch.cat([state, action], dim=2)
        
        Q_value = self.network(x)
        
        return Q_value


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

      
def sample_from_expbufCRITIC(experience_buffer, batch_size):
    
    perm_batch = np.random.permutation(len(experience_buffer))[:batch_size]
    
    experience = np.array(experience_buffer)[perm_batch]
    
    return experience[:,0], experience[:,1], experience[:,2], experience[:,3], experience[:,4], experience[:,5]        
    


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
    eps_decay_steps = 15000 
    
    global_step = 0  
    copy_steps = 100 
    start_steps = 1000
    steps_train = 4  
        
    batch_size = 32     
    
    n_episodes = 301
    
    gamma = 0.99 
    
    alpha = 0.01
    
    alpha_critic = 0.01 
    
    ###########################################################################  
  
    
    experience_bufferCRITIC = deque(maxlen=BUF_LEN)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    qofa_out = n_actions
        
    critic_network = MADDPG_Critic(obs_size*n_agents, n_agents).to(device)
    
    tgtCritic_network = MADDPG_Critic(obs_size*n_agents, n_agents).to(device)
    
    tgtCritic_network.load_state_dict(critic_network.state_dict())
    
    optimizerCritic = optim.Adam(params=critic_network.parameters(), lr=alpha_critic)
    
    objectiveCritic = nn.MSELoss()
    
    aq_network_list = []
    tgt_network_list = []
    optimizer_list = []
    objective_list = []
    exp_buf_L = []
    
    for agent_id in range(n_agents):
        
        actor_network = MADDPG_Actor(obs_size, n_actions).to(device)
        
        tgtActor_network = MADDPG_Actor(obs_size, n_actions).to(device)
       
        exp_buf_L.append(Exp_Buf()) 
        
        aq_network_list.append(actor_network)
        
        tgt_network_list.append(tgtActor_network)
        
        optimizer_list.append(optim.Adam(params=aq_network_list[agent_id].parameters(), lr=alpha))
        
        objective_list.append(nn.MSELoss())
    
    print ('aq_network_list[0]=', aq_network_list[0])
    print ('Critic_network_list=', critic_network)
    
    
    #####################################################
    for e in range(n_episodes):
        
        env.reset()
        
        terminated = False
        
        episode_reward = 0
        
        epsilon = max(eps_min, eps_max - (eps_max-eps_min) * global_step/eps_decay_steps)
        print ('epsilon=',epsilon)
               
        
        ##########################################################
        while not terminated:
                      
           
            actions = []
            action = 0
            observations = []
           
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
                
                for i in range(obs_size):
                    observations.append(obs_agent[agent_id][i])
            ############

           
            reward, terminated, _ = env.step(actions)
           
            episode_reward += reward
            
            
            actions_next = []
            observations_next = []
            
           
            if terminated == False:
                #####################################
                for agent_id in range(n_agents):
                    
                    obs_agent_next[agent_id] = env.get_obs_agent(agent_id)
                    
                    for i in range(obs_size):
                        observations_next.append(obs_agent_next[agent_id][i])
                
                    
                    pher_reinf = reward
                    
                    obs_agent_nextT = torch.FloatTensor([obs_agent_next[agent_id]]).to(device)
                    
                    action_probabilitiesT = tgt_network_list[agent_id](obs_agent_nextT)
                    
                    action_probabilitiesT = action_probabilitiesT.to("cpu")
                    action_probabilities = action_probabilitiesT.data.numpy()[0]
                    
                    avail_actions = env.get_avail_agent_actions(agent_id)
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    
                    action = select_actionFox(action_probabilities, avail_actions_ind, epsilon)
                    if action is None: action = np.random.choice (avail_actions_ind)
                    
                    
                    actions_next.append(action)
                #####################################
            elif terminated == True:
               
                actions_next = actions
                observations_next = observations
           
            
            experience_bufferCRITIC.append([observations, actions, observations_next, actions_next, reward, terminated])
                            
                
           
            ###########################################
            if (global_step % steps_train == 0) and (global_step > start_steps):
               
                exp_obs, exp_acts, exp_next_obs, exp_next_acts, exp_rew, exp_termd = sample_from_expbufCRITIC(experience_bufferCRITIC, batch_size)
                
                
                exp_obs = [x for x in exp_obs]
                obs_agentsT = torch.FloatTensor([exp_obs]).to(device)
                exp_acts = [x for x in exp_acts]
                act_agentsT = torch.FloatTensor([exp_acts]).to(device)
                
                ###############################
                
                action_probabilitieQT = critic_network(obs_agentsT, act_agentsT)
                action_probabilitieQT = action_probabilitieQT.to("cpu")
                               
               
                exp_next_obs = [x for x in exp_next_obs]
                obs_agents_nextT = torch.FloatTensor([exp_next_obs]).to(device)
                exp_next_acts = [x for x in exp_next_acts]
                act_agents_nextT = torch.FloatTensor([exp_next_acts]).to(device)
                                        
                
                action_probabilitieQ_nextT = tgtCritic_network(obs_agents_nextT, act_agents_nextT)
                
                action_probabilitieQ_nextT = action_probabilitieQ_nextT.to("cpu")
                action_probabilitieQ_next = action_probabilitieQ_nextT.data.numpy()[0]
                    
                
                y_batch = np.zeros([batch_size])
                action_probabilitieQBT = torch.empty(1, batch_size, dtype=torch.float)
                
                for i in range (batch_size):
                    
                    y_batch[i] = exp_rew[i] + (gamma*action_probabilitieQ_next[i])*(1 - exp_termd[i])
                    action_probabilitieQBT[0][i] = action_probabilitieQT[0][i]
                
                y_batchT = torch.FloatTensor([y_batch])
                
               
                optimizerCritic.zero_grad()
                 
                
                loss_t_critic = objectiveCritic(action_probabilitieQBT, y_batchT) 
                                  
                
                loss_t_critic.backward()
                
                optimizerCritic.step()
                ####################################
                                
                obs_local = np.zeros([n_agents, batch_size, obs_size])
               
                
                for agent_id in range(n_agents):
                    for i in range (batch_size):
                        for j in range (obs_size):
                            obs_local[agent_id][i][j] = exp_obs[i][j]
                            
                            
                obs_agentT0 = torch.FloatTensor([obs_local[0]]).to(device)
                obs_agentT1 = torch.FloatTensor([obs_local[1]]).to(device)
                obs_agentT2 = torch.FloatTensor([obs_local[2]]).to(device)
                obs_agentT3 = torch.FloatTensor([obs_local[3]]).to(device)
                obs_agentT4 = torch.FloatTensor([obs_local[4]]).to(device)
                obs_agentT5 = torch.FloatTensor([obs_local[5]]).to(device)
                obs_agentT6 = torch.FloatTensor([obs_local[6]]).to(device)
                obs_agentT7 = torch.FloatTensor([obs_local[7]]).to(device)
                obs_agentT8 = torch.FloatTensor([obs_local[8]]).to(device)
                obs_agentT9 = torch.FloatTensor([obs_local[9]]).to(device)
                obs_agentT10 = torch.FloatTensor([obs_local[10]]).to(device)
                obs_agentT11 = torch.FloatTensor([obs_local[11]]).to(device)
                obs_agentT12 = torch.FloatTensor([obs_local[12]]).to(device)
                obs_agentT13 = torch.FloatTensor([obs_local[13]]).to(device)
                obs_agentT14 = torch.FloatTensor([obs_local[14]]).to(device)
                obs_agentT15 = torch.FloatTensor([obs_local[15]]).to(device)
                obs_agentT16 = torch.FloatTensor([obs_local[16]]).to(device)
                obs_agentT17 = torch.FloatTensor([obs_local[17]]).to(device)
                obs_agentT18 = torch.FloatTensor([obs_local[18]]).to(device)
                obs_agentT19 = torch.FloatTensor([obs_local[19]]).to(device)
                obs_agentT20 = torch.FloatTensor([obs_local[20]]).to(device)
                obs_agentT21 = torch.FloatTensor([obs_local[21]]).to(device)
                obs_agentT22 = torch.FloatTensor([obs_local[22]]).to(device)
                obs_agentT23 = torch.FloatTensor([obs_local[23]]).to(device)
                obs_agentT24 = torch.FloatTensor([obs_local[24]]).to(device)
                obs_agentT25 = torch.FloatTensor([obs_local[25]]).to(device)
                obs_agentT26 = torch.FloatTensor([obs_local[26]]).to(device)
                obs_agentT27 = torch.FloatTensor([obs_local[27]]).to(device)
                obs_agentT28 = torch.FloatTensor([obs_local[28]]).to(device)
                obs_agentT29 = torch.FloatTensor([obs_local[29]]).to(device)
                obs_agentT30 = torch.FloatTensor([obs_local[30]]).to(device)
                obs_agentT31 = torch.FloatTensor([obs_local[31]]).to(device)
                obs_agentT32 = torch.FloatTensor([obs_local[32]]).to(device)
                obs_agentT33 = torch.FloatTensor([obs_local[33]]).to(device)
                obs_agentT34 = torch.FloatTensor([obs_local[34]]).to(device)
                obs_agentT35 = torch.FloatTensor([obs_local[35]]).to(device)
                obs_agentT36 = torch.FloatTensor([obs_local[36]]).to(device)
                obs_agentT37 = torch.FloatTensor([obs_local[37]]).to(device)
                obs_agentT38 = torch.FloatTensor([obs_local[38]]).to(device)
                obs_agentT39 = torch.FloatTensor([obs_local[39]]).to(device)
                obs_agentT40 = torch.FloatTensor([obs_local[40]]).to(device)
                obs_agentT41 = torch.FloatTensor([obs_local[41]]).to(device)
                obs_agentT42 = torch.FloatTensor([obs_local[42]]).to(device)
                obs_agentT43 = torch.FloatTensor([obs_local[43]]).to(device)
                obs_agentT44 = torch.FloatTensor([obs_local[44]]).to(device)
                obs_agentT45 = torch.FloatTensor([obs_local[45]]).to(device)
                obs_agentT46 = torch.FloatTensor([obs_local[46]]).to(device)
                obs_agentT47 = torch.FloatTensor([obs_local[47]]).to(device)
                obs_agentT48 = torch.FloatTensor([obs_local[48]]).to(device)
                obs_agentT49 = torch.FloatTensor([obs_local[49]]).to(device)
                obs_agentT50 = torch.FloatTensor([obs_local[50]]).to(device)
                obs_agentT51 = torch.FloatTensor([obs_local[51]]).to(device)
                obs_agentT52 = torch.FloatTensor([obs_local[52]]).to(device)
                obs_agentT53 = torch.FloatTensor([obs_local[53]]).to(device)
                obs_agentT54 = torch.FloatTensor([obs_local[54]]).to(device)
                obs_agentT55 = torch.FloatTensor([obs_local[55]]).to(device)
                obs_agentT56 = torch.FloatTensor([obs_local[56]]).to(device)
                obs_agentT57 = torch.FloatTensor([obs_local[57]]).to(device)
                obs_agentT58 = torch.FloatTensor([obs_local[58]]).to(device)
                obs_agentT59 = torch.FloatTensor([obs_local[59]]).to(device)
                obs_agentT60 = torch.FloatTensor([obs_local[60]]).to(device)
                obs_agentT61 = torch.FloatTensor([obs_local[61]]).to(device)
                obs_agentT62 = torch.FloatTensor([obs_local[62]]).to(device)
                obs_agentT63 = torch.FloatTensor([obs_local[63]]).to(device)
                obs_agentT64 = torch.FloatTensor([obs_local[64]]).to(device)
                obs_agentT65 = torch.FloatTensor([obs_local[65]]).to(device)
                obs_agentT66 = torch.FloatTensor([obs_local[66]]).to(device)
                obs_agentT67 = torch.FloatTensor([obs_local[67]]).to(device)
                obs_agentT68 = torch.FloatTensor([obs_local[68]]).to(device)
                obs_agentT69 = torch.FloatTensor([obs_local[69]]).to(device)
                obs_agentT70 = torch.FloatTensor([obs_local[70]]).to(device)
                obs_agentT71 = torch.FloatTensor([obs_local[71]]).to(device)
                obs_agentT72 = torch.FloatTensor([obs_local[72]]).to(device)
                obs_agentT73 = torch.FloatTensor([obs_local[73]]).to(device)
                obs_agentT74 = torch.FloatTensor([obs_local[74]]).to(device)
                
               
                optimizer_list[0].zero_grad()
                optimizer_list[1].zero_grad()
                optimizer_list[2].zero_grad()
                optimizer_list[3].zero_grad()
                optimizer_list[4].zero_grad()
                optimizer_list[5].zero_grad()
                optimizer_list[6].zero_grad()
                optimizer_list[7].zero_grad()
                optimizer_list[8].zero_grad()
                optimizer_list[9].zero_grad()
                optimizer_list[10].zero_grad()
                optimizer_list[11].zero_grad()
                optimizer_list[12].zero_grad()
                optimizer_list[13].zero_grad()
                optimizer_list[14].zero_grad()
                optimizer_list[15].zero_grad()
                optimizer_list[16].zero_grad()
                optimizer_list[17].zero_grad()
                optimizer_list[18].zero_grad()
                optimizer_list[19].zero_grad()
                optimizer_list[20].zero_grad()
                optimizer_list[21].zero_grad()
                optimizer_list[22].zero_grad()
                optimizer_list[23].zero_grad()
                optimizer_list[24].zero_grad()
                optimizer_list[25].zero_grad()
                optimizer_list[26].zero_grad()
                optimizer_list[27].zero_grad()
                optimizer_list[28].zero_grad()
                optimizer_list[29].zero_grad()
                optimizer_list[30].zero_grad()
                optimizer_list[31].zero_grad()
                optimizer_list[32].zero_grad()
                optimizer_list[33].zero_grad()
                optimizer_list[34].zero_grad()
                optimizer_list[35].zero_grad()
                optimizer_list[36].zero_grad()
                optimizer_list[37].zero_grad()
                optimizer_list[38].zero_grad()
                optimizer_list[39].zero_grad()
                optimizer_list[40].zero_grad()
                optimizer_list[41].zero_grad()
                optimizer_list[42].zero_grad()
                optimizer_list[43].zero_grad()
                optimizer_list[44].zero_grad()
                optimizer_list[45].zero_grad()
                optimizer_list[46].zero_grad()
                optimizer_list[47].zero_grad()
                optimizer_list[48].zero_grad()
                optimizer_list[49].zero_grad()
                optimizer_list[50].zero_grad()
                optimizer_list[51].zero_grad()
                optimizer_list[52].zero_grad()
                optimizer_list[53].zero_grad()
                optimizer_list[54].zero_grad()
                optimizer_list[55].zero_grad()
                optimizer_list[56].zero_grad()
                optimizer_list[57].zero_grad()
                optimizer_list[58].zero_grad()
                optimizer_list[59].zero_grad()
                optimizer_list[60].zero_grad()
                optimizer_list[61].zero_grad()
                optimizer_list[62].zero_grad()
                optimizer_list[63].zero_grad()
                optimizer_list[64].zero_grad()
                optimizer_list[65].zero_grad()
                optimizer_list[66].zero_grad()
                optimizer_list[67].zero_grad()
                optimizer_list[68].zero_grad()
                optimizer_list[69].zero_grad()
                optimizer_list[70].zero_grad()
                optimizer_list[71].zero_grad()
                optimizer_list[72].zero_grad()
                optimizer_list[73].zero_grad()
                optimizer_list[74].zero_grad()
                                
                
                action_probabilitiesT0 = aq_network_list[0](obs_agentT0)
                action_probabilitiesT1 = aq_network_list[1](obs_agentT1)
                action_probabilitiesT2 = aq_network_list[2](obs_agentT2)
                action_probabilitiesT3 = aq_network_list[3](obs_agentT3)
                action_probabilitiesT4 = aq_network_list[4](obs_agentT4)
                action_probabilitiesT5 = aq_network_list[5](obs_agentT5)
                action_probabilitiesT6 = aq_network_list[6](obs_agentT6)
                action_probabilitiesT7 = aq_network_list[7](obs_agentT7)
                action_probabilitiesT8 = aq_network_list[8](obs_agentT8)
                action_probabilitiesT9 = aq_network_list[9](obs_agentT9)
                action_probabilitiesT10 = aq_network_list[10](obs_agentT10)
                action_probabilitiesT11 = aq_network_list[11](obs_agentT11)
                action_probabilitiesT12 = aq_network_list[12](obs_agentT12)
                action_probabilitiesT13 = aq_network_list[13](obs_agentT13)
                action_probabilitiesT14 = aq_network_list[14](obs_agentT14)
                action_probabilitiesT15 = aq_network_list[15](obs_agentT15)
                action_probabilitiesT16 = aq_network_list[16](obs_agentT16)
                action_probabilitiesT17 = aq_network_list[17](obs_agentT17)
                action_probabilitiesT18 = aq_network_list[18](obs_agentT18)
                action_probabilitiesT19 = aq_network_list[19](obs_agentT19)
                action_probabilitiesT20 = aq_network_list[20](obs_agentT20)
                action_probabilitiesT21 = aq_network_list[21](obs_agentT21)
                action_probabilitiesT22 = aq_network_list[22](obs_agentT22)
                action_probabilitiesT23 = aq_network_list[23](obs_agentT23)
                action_probabilitiesT24 = aq_network_list[24](obs_agentT24)
                action_probabilitiesT25 = aq_network_list[25](obs_agentT25)
                action_probabilitiesT26 = aq_network_list[26](obs_agentT26)
                action_probabilitiesT27 = aq_network_list[27](obs_agentT27)
                action_probabilitiesT28 = aq_network_list[28](obs_agentT28)
                action_probabilitiesT29 = aq_network_list[29](obs_agentT29)
                action_probabilitiesT30 = aq_network_list[30](obs_agentT30)
                action_probabilitiesT31 = aq_network_list[31](obs_agentT31)
                action_probabilitiesT32 = aq_network_list[32](obs_agentT32)
                action_probabilitiesT33 = aq_network_list[33](obs_agentT33)
                action_probabilitiesT34 = aq_network_list[34](obs_agentT34)
                action_probabilitiesT35 = aq_network_list[35](obs_agentT35)
                action_probabilitiesT36 = aq_network_list[36](obs_agentT36)
                action_probabilitiesT37 = aq_network_list[37](obs_agentT37)
                action_probabilitiesT38 = aq_network_list[38](obs_agentT38)
                action_probabilitiesT39 = aq_network_list[39](obs_agentT39)
                action_probabilitiesT40 = aq_network_list[40](obs_agentT40)
                action_probabilitiesT41 = aq_network_list[41](obs_agentT41)
                action_probabilitiesT42 = aq_network_list[42](obs_agentT42)
                action_probabilitiesT43 = aq_network_list[43](obs_agentT43)
                action_probabilitiesT44 = aq_network_list[44](obs_agentT44)
                action_probabilitiesT45 = aq_network_list[45](obs_agentT45)
                action_probabilitiesT46 = aq_network_list[46](obs_agentT46)
                action_probabilitiesT47 = aq_network_list[47](obs_agentT47)
                action_probabilitiesT48 = aq_network_list[48](obs_agentT48)
                action_probabilitiesT49 = aq_network_list[49](obs_agentT49)
                action_probabilitiesT50 = aq_network_list[50](obs_agentT50)
                action_probabilitiesT51 = aq_network_list[51](obs_agentT51)
                action_probabilitiesT52 = aq_network_list[52](obs_agentT52)
                action_probabilitiesT53 = aq_network_list[53](obs_agentT53)
                action_probabilitiesT54 = aq_network_list[54](obs_agentT54)
                action_probabilitiesT55 = aq_network_list[55](obs_agentT55)
                action_probabilitiesT56 = aq_network_list[56](obs_agentT56)
                action_probabilitiesT57 = aq_network_list[57](obs_agentT57)
                action_probabilitiesT58 = aq_network_list[58](obs_agentT58)
                action_probabilitiesT59 = aq_network_list[59](obs_agentT59)
                action_probabilitiesT60 = aq_network_list[60](obs_agentT60)
                action_probabilitiesT61 = aq_network_list[61](obs_agentT61)
                action_probabilitiesT62 = aq_network_list[62](obs_agentT62)
                action_probabilitiesT63 = aq_network_list[63](obs_agentT63)
                action_probabilitiesT64 = aq_network_list[64](obs_agentT64)
                action_probabilitiesT65 = aq_network_list[65](obs_agentT65)
                action_probabilitiesT66 = aq_network_list[66](obs_agentT66)
                action_probabilitiesT67 = aq_network_list[67](obs_agentT67)
                action_probabilitiesT68 = aq_network_list[68](obs_agentT68)
                action_probabilitiesT69 = aq_network_list[69](obs_agentT69)
                action_probabilitiesT70 = aq_network_list[70](obs_agentT70)
                action_probabilitiesT71 = aq_network_list[71](obs_agentT71)
                action_probabilitiesT72 = aq_network_list[72](obs_agentT72)
                action_probabilitiesT73 = aq_network_list[73](obs_agentT73)
                action_probabilitiesT74 = aq_network_list[74](obs_agentT74)
                                                
               
                action_probabilitiesT0 = action_probabilitiesT0.to("cpu")
                action_probabilitiesT1 = action_probabilitiesT1.to("cpu")
                action_probabilitiesT2 = action_probabilitiesT2.to("cpu")
                action_probabilitiesT3 = action_probabilitiesT3.to("cpu")
                action_probabilitiesT4 = action_probabilitiesT4.to("cpu")
                action_probabilitiesT5 = action_probabilitiesT5.to("cpu")
                action_probabilitiesT6 = action_probabilitiesT6.to("cpu")
                action_probabilitiesT7 = action_probabilitiesT7.to("cpu")
                action_probabilitiesT8 = action_probabilitiesT8.to("cpu")
                action_probabilitiesT9 = action_probabilitiesT9.to("cpu")
                action_probabilitiesT10 = action_probabilitiesT10.to("cpu")
                action_probabilitiesT11 = action_probabilitiesT11.to("cpu")
                action_probabilitiesT12 = action_probabilitiesT12.to("cpu")
                action_probabilitiesT13 = action_probabilitiesT13.to("cpu")
                action_probabilitiesT14 = action_probabilitiesT14.to("cpu")
                action_probabilitiesT15 = action_probabilitiesT15.to("cpu")
                action_probabilitiesT16 = action_probabilitiesT16.to("cpu")
                action_probabilitiesT17 = action_probabilitiesT17.to("cpu")
                action_probabilitiesT18 = action_probabilitiesT18.to("cpu")
                action_probabilitiesT19 = action_probabilitiesT19.to("cpu")
                action_probabilitiesT20 = action_probabilitiesT20.to("cpu")
                action_probabilitiesT21 = action_probabilitiesT21.to("cpu")
                action_probabilitiesT22 = action_probabilitiesT22.to("cpu")
                action_probabilitiesT23 = action_probabilitiesT23.to("cpu")
                action_probabilitiesT24 = action_probabilitiesT24.to("cpu")
                action_probabilitiesT25 = action_probabilitiesT25.to("cpu")
                action_probabilitiesT26 = action_probabilitiesT26.to("cpu")
                action_probabilitiesT27 = action_probabilitiesT27.to("cpu")
                action_probabilitiesT28 = action_probabilitiesT28.to("cpu")
                action_probabilitiesT29 = action_probabilitiesT29.to("cpu")
                action_probabilitiesT30 = action_probabilitiesT30.to("cpu")
                action_probabilitiesT31 = action_probabilitiesT31.to("cpu")
                action_probabilitiesT32 = action_probabilitiesT32.to("cpu")
                action_probabilitiesT33 = action_probabilitiesT33.to("cpu")
                action_probabilitiesT34 = action_probabilitiesT34.to("cpu")
                action_probabilitiesT35 = action_probabilitiesT35.to("cpu")
                action_probabilitiesT36 = action_probabilitiesT36.to("cpu")
                action_probabilitiesT37 = action_probabilitiesT37.to("cpu")
                action_probabilitiesT38 = action_probabilitiesT38.to("cpu")
                action_probabilitiesT39 = action_probabilitiesT39.to("cpu")
                action_probabilitiesT40 = action_probabilitiesT40.to("cpu")
                action_probabilitiesT41 = action_probabilitiesT41.to("cpu")
                action_probabilitiesT42 = action_probabilitiesT42.to("cpu")
                action_probabilitiesT43 = action_probabilitiesT43.to("cpu")
                action_probabilitiesT44 = action_probabilitiesT44.to("cpu")
                action_probabilitiesT45 = action_probabilitiesT45.to("cpu")
                action_probabilitiesT46 = action_probabilitiesT46.to("cpu")
                action_probabilitiesT47 = action_probabilitiesT47.to("cpu")
                action_probabilitiesT48 = action_probabilitiesT48.to("cpu")
                action_probabilitiesT49 = action_probabilitiesT49.to("cpu")
                action_probabilitiesT50 = action_probabilitiesT50.to("cpu")
                action_probabilitiesT51 = action_probabilitiesT51.to("cpu")
                action_probabilitiesT52 = action_probabilitiesT52.to("cpu")
                action_probabilitiesT53 = action_probabilitiesT53.to("cpu")
                action_probabilitiesT54 = action_probabilitiesT54.to("cpu")
                action_probabilitiesT55 = action_probabilitiesT55.to("cpu")
                action_probabilitiesT56 = action_probabilitiesT56.to("cpu")
                action_probabilitiesT57 = action_probabilitiesT57.to("cpu")
                action_probabilitiesT58 = action_probabilitiesT58.to("cpu")
                action_probabilitiesT59 = action_probabilitiesT59.to("cpu")
                action_probabilitiesT60 = action_probabilitiesT60.to("cpu")
                action_probabilitiesT61 = action_probabilitiesT61.to("cpu")
                action_probabilitiesT62 = action_probabilitiesT62.to("cpu")
                action_probabilitiesT63 = action_probabilitiesT63.to("cpu")
                action_probabilitiesT64 = action_probabilitiesT64.to("cpu")
                action_probabilitiesT65 = action_probabilitiesT65.to("cpu")
                action_probabilitiesT66 = action_probabilitiesT66.to("cpu")
                action_probabilitiesT67 = action_probabilitiesT67.to("cpu")
                action_probabilitiesT68 = action_probabilitiesT68.to("cpu")
                action_probabilitiesT69 = action_probabilitiesT69.to("cpu")
                action_probabilitiesT70 = action_probabilitiesT70.to("cpu")
                action_probabilitiesT71 = action_probabilitiesT71.to("cpu")
                action_probabilitiesT72 = action_probabilitiesT72.to("cpu")
                action_probabilitiesT73 = action_probabilitiesT73.to("cpu")
                action_probabilitiesT74 = action_probabilitiesT74.to("cpu")
                
                action_probabilities0 = action_probabilitiesT0.data.numpy()[0]
                action_probabilities1 = action_probabilitiesT1.data.numpy()[0]
                action_probabilities2 = action_probabilitiesT2.data.numpy()[0]
                action_probabilities3 = action_probabilitiesT3.data.numpy()[0]
                action_probabilities4 = action_probabilitiesT4.data.numpy()[0]
                action_probabilities5 = action_probabilitiesT5.data.numpy()[0]
                action_probabilities6 = action_probabilitiesT6.data.numpy()[0]
                action_probabilities7 = action_probabilitiesT7.data.numpy()[0]
                action_probabilities8 = action_probabilitiesT8.data.numpy()[0]
                action_probabilities9 = action_probabilitiesT9.data.numpy()[0]
                action_probabilities10 = action_probabilitiesT10.data.numpy()[0]
                action_probabilities11 = action_probabilitiesT11.data.numpy()[0]
                action_probabilities12 = action_probabilitiesT12.data.numpy()[0]
                action_probabilities13 = action_probabilitiesT13.data.numpy()[0]
                action_probabilities14 = action_probabilitiesT14.data.numpy()[0]
                action_probabilities15 = action_probabilitiesT15.data.numpy()[0]
                action_probabilities16 = action_probabilitiesT16.data.numpy()[0]
                action_probabilities17 = action_probabilitiesT17.data.numpy()[0]
                action_probabilities18 = action_probabilitiesT18.data.numpy()[0]
                action_probabilities19 = action_probabilitiesT19.data.numpy()[0]
                action_probabilities20 = action_probabilitiesT20.data.numpy()[0]
                action_probabilities21 = action_probabilitiesT21.data.numpy()[0]
                action_probabilities22 = action_probabilitiesT22.data.numpy()[0]
                action_probabilities23 = action_probabilitiesT23.data.numpy()[0]
                action_probabilities24 = action_probabilitiesT24.data.numpy()[0]
                action_probabilities25 = action_probabilitiesT25.data.numpy()[0]
                action_probabilities26 = action_probabilitiesT26.data.numpy()[0]
                action_probabilities27 = action_probabilitiesT27.data.numpy()[0]
                action_probabilities28 = action_probabilitiesT28.data.numpy()[0]
                action_probabilities29 = action_probabilitiesT29.data.numpy()[0]
                action_probabilities30 = action_probabilitiesT30.data.numpy()[0]
                action_probabilities31 = action_probabilitiesT31.data.numpy()[0]
                action_probabilities32 = action_probabilitiesT32.data.numpy()[0]
                action_probabilities33 = action_probabilitiesT33.data.numpy()[0]
                action_probabilities34 = action_probabilitiesT34.data.numpy()[0]
                action_probabilities35 = action_probabilitiesT35.data.numpy()[0]
                action_probabilities36 = action_probabilitiesT36.data.numpy()[0]
                action_probabilities37 = action_probabilitiesT37.data.numpy()[0]
                action_probabilities38 = action_probabilitiesT38.data.numpy()[0]
                action_probabilities39 = action_probabilitiesT39.data.numpy()[0]
                action_probabilities40 = action_probabilitiesT40.data.numpy()[0]
                action_probabilities41 = action_probabilitiesT41.data.numpy()[0]
                action_probabilities42 = action_probabilitiesT42.data.numpy()[0]
                action_probabilities43 = action_probabilitiesT43.data.numpy()[0]
                action_probabilities44 = action_probabilitiesT44.data.numpy()[0]
                action_probabilities45 = action_probabilitiesT45.data.numpy()[0]
                action_probabilities46 = action_probabilitiesT46.data.numpy()[0]
                action_probabilities47 = action_probabilitiesT47.data.numpy()[0]
                action_probabilities48 = action_probabilitiesT48.data.numpy()[0]
                action_probabilities49 = action_probabilitiesT49.data.numpy()[0]
                action_probabilities50 = action_probabilitiesT50.data.numpy()[0]
                action_probabilities51 = action_probabilitiesT51.data.numpy()[0]
                action_probabilities52 = action_probabilitiesT52.data.numpy()[0]
                action_probabilities53 = action_probabilitiesT53.data.numpy()[0]
                action_probabilities54 = action_probabilitiesT54.data.numpy()[0]
                action_probabilities55 = action_probabilitiesT55.data.numpy()[0]
                action_probabilities56 = action_probabilitiesT56.data.numpy()[0]
                action_probabilities57 = action_probabilitiesT57.data.numpy()[0]
                action_probabilities58 = action_probabilitiesT58.data.numpy()[0]
                action_probabilities59 = action_probabilitiesT59.data.numpy()[0]
                action_probabilities60 = action_probabilitiesT60.data.numpy()[0]
                action_probabilities61 = action_probabilitiesT61.data.numpy()[0]
                action_probabilities62 = action_probabilitiesT62.data.numpy()[0]
                action_probabilities63 = action_probabilitiesT63.data.numpy()[0]
                action_probabilities64 = action_probabilitiesT64.data.numpy()[0]
                action_probabilities65 = action_probabilitiesT65.data.numpy()[0]
                action_probabilities66 = action_probabilitiesT66.data.numpy()[0]
                action_probabilities67 = action_probabilitiesT67.data.numpy()[0]
                action_probabilities68 = action_probabilitiesT68.data.numpy()[0]
                action_probabilities69 = action_probabilitiesT69.data.numpy()[0]
                action_probabilities70 = action_probabilitiesT70.data.numpy()[0]
                action_probabilities71 = action_probabilitiesT71.data.numpy()[0]
                action_probabilities72 = action_probabilitiesT72.data.numpy()[0]
                action_probabilities73 = action_probabilitiesT73.data.numpy()[0]
                action_probabilities74 = action_probabilitiesT74.data.numpy()[0]
                
                
                act_full = np.zeros([batch_size, n_agents])
                for i in range (batch_size):
                    act_full[i][0] = np.argmax(action_probabilities0[i])
                    act_full[i][1] = np.argmax(action_probabilities1[i])
                    act_full[i][2] = np.argmax(action_probabilities2[i])
                    act_full[i][3] = np.argmax(action_probabilities3[i])
                    act_full[i][4] = np.argmax(action_probabilities4[i])
                    act_full[i][5] = np.argmax(action_probabilities5[i])
                    act_full[i][6] = np.argmax(action_probabilities6[i])
                    act_full[i][7] = np.argmax(action_probabilities7[i])
                    act_full[i][8] = np.argmax(action_probabilities8[i])
                    act_full[i][9] = np.argmax(action_probabilities9[i])
                    act_full[i][10] = np.argmax(action_probabilities10[i])
                    act_full[i][11] = np.argmax(action_probabilities11[i])
                    act_full[i][12] = np.argmax(action_probabilities12[i])
                    act_full[i][13] = np.argmax(action_probabilities13[i])
                    act_full[i][14] = np.argmax(action_probabilities14[i])
                    act_full[i][15] = np.argmax(action_probabilities15[i])
                    act_full[i][16] = np.argmax(action_probabilities16[i])
                    act_full[i][17] = np.argmax(action_probabilities17[i])
                    act_full[i][18] = np.argmax(action_probabilities18[i])
                    act_full[i][19] = np.argmax(action_probabilities19[i])
                    act_full[i][20] = np.argmax(action_probabilities20[i])
                    act_full[i][21] = np.argmax(action_probabilities21[i])
                    act_full[i][22] = np.argmax(action_probabilities22[i])
                    act_full[i][23] = np.argmax(action_probabilities23[i])
                    act_full[i][24] = np.argmax(action_probabilities24[i])
                    act_full[i][25] = np.argmax(action_probabilities25[i])
                    act_full[i][26] = np.argmax(action_probabilities26[i])
                    act_full[i][27] = np.argmax(action_probabilities27[i])
                    act_full[i][28] = np.argmax(action_probabilities28[i])
                    act_full[i][29] = np.argmax(action_probabilities29[i])
                    act_full[i][30] = np.argmax(action_probabilities30[i])
                    act_full[i][31] = np.argmax(action_probabilities31[i])
                    act_full[i][32] = np.argmax(action_probabilities32[i])
                    act_full[i][33] = np.argmax(action_probabilities33[i])
                    act_full[i][34] = np.argmax(action_probabilities34[i])
                    act_full[i][35] = np.argmax(action_probabilities35[i])
                    act_full[i][36] = np.argmax(action_probabilities36[i])
                    act_full[i][37] = np.argmax(action_probabilities37[i])
                    act_full[i][38] = np.argmax(action_probabilities38[i])
                    act_full[i][39] = np.argmax(action_probabilities39[i])
                    act_full[i][40] = np.argmax(action_probabilities40[i])
                    act_full[i][41] = np.argmax(action_probabilities41[i])
                    act_full[i][42] = np.argmax(action_probabilities42[i])
                    act_full[i][43] = np.argmax(action_probabilities43[i])
                    act_full[i][44] = np.argmax(action_probabilities44[i])
                    act_full[i][45] = np.argmax(action_probabilities45[i])
                    act_full[i][46] = np.argmax(action_probabilities46[i])
                    act_full[i][47] = np.argmax(action_probabilities47[i])
                    act_full[i][48] = np.argmax(action_probabilities48[i])
                    act_full[i][49] = np.argmax(action_probabilities49[i])
                    act_full[i][50] = np.argmax(action_probabilities50[i])
                    act_full[i][51] = np.argmax(action_probabilities51[i])
                    act_full[i][52] = np.argmax(action_probabilities52[i])
                    act_full[i][53] = np.argmax(action_probabilities53[i])
                    act_full[i][54] = np.argmax(action_probabilities54[i])
                    act_full[i][55] = np.argmax(action_probabilities55[i])
                    act_full[i][56] = np.argmax(action_probabilities56[i])
                    act_full[i][57] = np.argmax(action_probabilities57[i])
                    act_full[i][58] = np.argmax(action_probabilities58[i])
                    act_full[i][59] = np.argmax(action_probabilities59[i])
                    act_full[i][60] = np.argmax(action_probabilities60[i])
                    act_full[i][61] = np.argmax(action_probabilities61[i])
                    act_full[i][62] = np.argmax(action_probabilities62[i])
                    act_full[i][63] = np.argmax(action_probabilities63[i])
                    act_full[i][64] = np.argmax(action_probabilities64[i])
                    act_full[i][65] = np.argmax(action_probabilities65[i])
                    act_full[i][66] = np.argmax(action_probabilities66[i])
                    act_full[i][67] = np.argmax(action_probabilities67[i])
                    act_full[i][68] = np.argmax(action_probabilities68[i])
                    act_full[i][69] = np.argmax(action_probabilities69[i])
                    act_full[i][70] = np.argmax(action_probabilities70[i])
                    act_full[i][71] = np.argmax(action_probabilities71[i])
                    act_full[i][72] = np.argmax(action_probabilities72[i])
                    act_full[i][73] = np.argmax(action_probabilities73[i])
                    act_full[i][74] = np.argmax(action_probabilities74[i])
                    
                act_fullT = torch.FloatTensor([act_full]).to(device)
                
                
                obs_agentsT = torch.FloatTensor([exp_obs]).to(device)
                                
               
                actor_lossT = -critic_network(obs_agentsT, act_fullT)
                
                
                actor_lossT = actor_lossT.mean()    
                
                
                actor_lossT.backward()
                
               
                optimizer_list[0].step()
                optimizer_list[1].step()
                optimizer_list[2].step()
                optimizer_list[3].step()
                optimizer_list[4].step()
                optimizer_list[5].step()
                optimizer_list[6].step()
                optimizer_list[7].step()
                optimizer_list[8].step()
                optimizer_list[9].step()
                optimizer_list[10].step()
                optimizer_list[11].step()
                optimizer_list[12].step()
                optimizer_list[13].step()
                optimizer_list[14].step()
                optimizer_list[15].step()
                optimizer_list[16].step()
                optimizer_list[17].step()
                optimizer_list[18].step()
                optimizer_list[19].step()
                optimizer_list[20].step()
                optimizer_list[21].step()
                optimizer_list[22].step()
                optimizer_list[23].step()
                optimizer_list[24].step()
                optimizer_list[25].step()
                optimizer_list[26].step()
                optimizer_list[27].step()
                optimizer_list[28].step()
                optimizer_list[29].step()
                optimizer_list[30].step()
                optimizer_list[31].step()
                optimizer_list[32].step()
                optimizer_list[33].step()
                optimizer_list[34].step()
                optimizer_list[35].step()
                optimizer_list[36].step()
                optimizer_list[37].step()
                optimizer_list[38].step()
                optimizer_list[39].step()
                optimizer_list[40].step()
                optimizer_list[41].step()
                optimizer_list[42].step()
                optimizer_list[43].step()
                optimizer_list[44].step()
                optimizer_list[45].step()
                optimizer_list[46].step()
                optimizer_list[47].step()
                optimizer_list[48].step()
                optimizer_list[49].step()
                optimizer_list[50].step()
                optimizer_list[51].step()
                optimizer_list[52].step()
                optimizer_list[53].step()
                optimizer_list[54].step()
                optimizer_list[55].step()
                optimizer_list[56].step()
                optimizer_list[57].step()
                optimizer_list[58].step()
                optimizer_list[59].step()
                optimizer_list[60].step()
                optimizer_list[61].step()
                optimizer_list[62].step()
                optimizer_list[63].step()
                optimizer_list[64].step()
                optimizer_list[65].step()
                optimizer_list[66].step()
                optimizer_list[67].step()
                optimizer_list[68].step()
                optimizer_list[69].step()
                optimizer_list[70].step()
                optimizer_list[71].step()
                optimizer_list[72].step()
                optimizer_list[73].step()
                optimizer_list[74].step()
                
                                  
               
                
            
            if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                tgtCritic_network.load_state_dict(critic_network.state_dict())
            
            ##############################
                  
            
            global_step += 1
                        
           
        ####################################################
        
        print('global_step=', global_step, "Total reward in episode {} = {}".format(e, episode_reward))
       
        
    ############################################################
    
   
    env.close()
   
    for agent_id in range(n_agents):
        torch.save(aq_network_list[agent_id].state_dict(),"aqnet_%.0f.dat"%agent_id) 
        
   

 
if __name__ == "__main__":
    start_time = time.time()
    main()
    
    print("--- %s минут ---" % ((time.time() - start_time)/60))