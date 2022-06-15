import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np
class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,trivial,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, gamma=0.95, tau=0.01):
        self.trivial = trivial  #True or False. If ture, the updating rule will be changed.
        self.gamma = gamma #dqn
        self.tau = tau#MADDPG软更新的步长
        self.id = agent_idx
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)#保证一开始的时候两种网络的参数保持一致
        
        
    def choose_action(self, observation,rewards,act):
        if self.trivial:
            """
            waiting for adjustment!!!!!!!!!!!!!!!
            """  
            actions = 0
            obs = observation.reshape(2,-1)
            j = np.random.choice(np.arange(1,obs.shape[1]))
            temp = 1 / (1+ np.exp(rewards - obs[1,j]) *2) 
            if np.random.rand() < temp:
                actions = obs[0,j]
            else:
                actions = act[0]
            
            ACTIONS = np.array([actions, 1-actions])
            # obs = observation.reshape(2,-1)
            # actions = obs[0,np.argmax( obs[1,:])]
            # ACTIONS = np.array([actions, 1-actions])
            
         
        else:
            obs = observation.reshape(2,-1)
            obs = obs[:, 1:]
            state = T.tensor([obs], dtype=T.float).to(self.actor.device).reshape(-1)
            
            actions = self.actor.forward(state)
            # noise = T.rand(self.n_actions).to(self.actor.device)
            # action = actions + noise
            #原本计划加入随机噪声，后来有bug
            ACTIONS = actions.detach().cpu().numpy().flatten()
        
        return ACTIONS

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
