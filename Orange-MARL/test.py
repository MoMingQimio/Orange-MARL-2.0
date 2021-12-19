import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from OrangeEnv import MultiAgentEnv



import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

scenario = "lattice"
n_agents = 4
env = MultiAgentEnv(n_agents, scenario)
actor_dims = []

#waiting for adjustment  
for i in range(n_agents):
    actor_dims.append(env.observation_space[i] * 2)
critic_dims = n_agents * 2

n_actions= 2

maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=20, fc2=30,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')


memory = MultiAgentReplayBuffer(1, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1)


PRINT_INTERVAL = 1
N_GAMES = 1
MAX_STEPS = 1
total_steps = 0
score_history = []
evaluate = False
best_score = 0


state, obs = env.reset()
score = 0
done = [False]*n_agents
episode_step = 0
for i in range(1):
    actions = maddpg_agents.choose_action(obs)
    state_, obs_, reward, done, info = env.step(np.array(actions)[:,0])
    memory.store_transition(obs, state, actions, reward, obs_, state_, done)


actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()
        
        
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
states = T.tensor(states, dtype=T.float32).to(device)
actions = T.tensor(actions, dtype=T.float32).to(device)
rewards = T.tensor(rewards).to(device)
states_ = T.tensor(states_, dtype=T.float32).to(device)
dones = T.tensor(dones).to(device)

all_agents_new_actions = []
all_agents_new_mu_actions = []
old_agents_actions = []


for agent_idx, agent in enumerate(maddpg_agents.agents):
    new_states = T.tensor(actor_new_states[agent_idx], 
                                    dtype=T.float).to(device)
    #actions = maddpg_agents.choose_action(obs)

    new_pi = maddpg_agents.agents[agent_idx].target_actor.forward(new_states)

    all_agents_new_actions.append(new_pi)
    mu_states = T.tensor(actor_states[agent_idx], 
                            dtype=T.float).to(device)
    pi = maddpg_agents.agents[agent_idx].actor.forward(mu_states)
    all_agents_new_mu_actions.append(pi)
    old_agents_actions.append(actions[agent_idx])
    
new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

agent_idx = 0
critic_value_ = maddpg_agents.agents[agent_idx].target_critic.forward(states_, new_actions).flatten()
#critic_value_[dones[:,0]] = 0.0
critic_value = maddpg_agents.agents[agent_idx].critic.forward(states, old_actions).flatten()

target = rewards[:,agent_idx] + maddpg_agents.agents[agent_idx].gamma*critic_value_
critic_loss = F.mse_loss(target.detach(), critic_value.double())
maddpg_agents.agents[agent_idx].critic.optimizer.zero_grad()
with T.autograd.detect_anomaly():
    critic_loss.backward(
        retain_graph=True
        )


actor_loss_ = maddpg_agents.agents[agent_idx].critic.forward(states, mu).flatten()
actor_loss = -T.mean(actor_loss_)
maddpg_agents.agents[agent_idx].actor.optimizer.zero_grad()
actor_loss.backward(retain_graph=True)
maddpg_agents.agents[agent_idx].actor.optimizer.step()
#maddpg_agents.agents[agent_idx].critic.optimizer.step()
# maddpg_agents.agents[agent_idx].update_network_parameters()

agent_idx = 1
critic_value_ = maddpg_agents.agents[agent_idx].target_critic.forward(states_, new_actions).flatten()
#critic_value_[dones[:,0]] = 0.0
critic_value = maddpg_agents.agents[agent_idx].critic.forward(states, old_actions).flatten()

target = rewards[:,agent_idx] + maddpg_agents.agents[agent_idx].gamma*critic_value_
critic_loss = F.mse_loss(target.detach(), critic_value.double())
maddpg_agents.agents[agent_idx].critic.optimizer.zero_grad()
with T.autograd.set_detect_anomaly(True):
    critic_loss.backward(
        retain_graph=True
        )
maddpg_agents.agents[agent_idx].critic.optimizer.step()

actor_loss_ = maddpg_agents.agents[agent_idx].critic.forward(states, mu).flatten()
actor_loss = -T.mean(actor_loss_)
maddpg_agents.agents[agent_idx].actor.optimizer.zero_grad()
with T.autograd.set_detect_anomaly(True):
    actor_loss.backward(retain_graph=True)
maddpg_agents.agents[agent_idx].actor.optimizer.step()

