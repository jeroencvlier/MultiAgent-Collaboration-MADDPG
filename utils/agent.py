import ujson
import random
import numpy as np
from IPython.display import clear_output

# agent libraries
from utils.actorCritic import ActorPolicy, CriticPolicy
from utils.per import PrioritizedExperienceReplay
from utils.noise import OrnsteinUhlenbeckNoise

# Plotting Modules
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent:
    '''Defines the agent that interacts and learns with the environment'''
    def __init__(self,num_agents,state_size,action_size, memory_size = 500000, replay_size = 1000, gamma = 0.95, tau=0.01, update_frequency = 20, learn_steps = 100, lr_actor = 0.0001, lr_critic = 0.001, seed = 333, print_every = 50):
        # Preset variables for learning
        self.tau = tau
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        self.memory_size = memory_size
        self.replay_size = replay_size
        self.update_frequency = update_frequency
        self.learn_steps = learn_steps
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.msgs = []
        
        self.print_every = print_every
        
        # initialize critic network Q(s,a|θQ) and actor µ(s|θµ) with weights θQ and θµ        
        self.actor = ActorPolicy(state_size,action_size)
        self.critic = CriticPolicy(state_size,action_size)
        
        # initiate optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Initialize target network Q and µ: θQ'← θQ, θµ'← θµ
        self.actor_target = ActorPolicy(self.state_size,self.action_size)
        self.critic_target = CriticPolicy(self.state_size,self.action_size)

        # copy parameters state dictionary
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # instantiate memory replay
        self.memory = PrioritizedExperienceReplay(memory_size,replay_size)

        # Initialize time step (for updating every update_frequency steps)
        self.freq_update = 0
        
        # Initate noise
        self.noise = OrnsteinUhlenbeckNoise((num_agents,action_size), seed)
        
        
    def soft_update(self, target_net,local_net,tau):
        '''Soft update: θ_target = τ * θ_local + (1 - τ) * θ_target
        target_net (pytorch)
        local_net
        tau            (float) : Soft update parameter
        '''
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        return target_net
    
    def act(self, env, brain_name, state, add_noise = True):
        '''Agent takes an action givern the state and returns a reward. 
        When the freq_update parameter is met then the agent will learn
        
        add_noise (Bool) : Adds noise to the action parameter
        '''
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        
        if add_noise == True:
            action += self.noise.sample()
            # clip after noise
            action = action.clip(-1,1)    
            
        env_info = env.step(action)[brain_name]
        done = env_info.local_done
        next_state = env_info.vector_observations
        reward = env_info.rewards
        
        na = self.actor_target(next_state)
        q_target_next = self.critic_target(next_state, na).detach()
        q_target = torch.FloatTensor(reward).view(-1,1) + (self.gamma * (1-torch.FloatTensor(done).view(-1,1)) * q_target_next) 
        q = self.critic(state, torch.FloatTensor(action))

        error = abs(q - q_target).detach().numpy().max(1)
        self.error = error
        
        # Store replay buffer
        for s_,a_,r_,ns_,d_,e_ in zip(state, action, reward, next_state, done,error):
            if torch.is_tensor(s_): s_ = s_.detach().numpy()
            if torch.is_tensor(a_): a_ = a_.detach().numpy()
            if torch.is_tensor(r_): r_ = r_.detach().numpy()
            if torch.is_tensor(ns_): ns_ = ns_.detach().numpy()
            if torch.is_tensor(d_): d_ = d_.detach().numpy()
            if torch.is_tensor(e_): e_ = e_.detach().numpy()
            self.memory.add(e_, (s_,a_,r_,ns_,d_))
            
        self.freq_update += 1
        if (self.memory.tree.n_entries > self.replay_size) and (self.freq_update%self.update_frequency==0):
            self.learn()
                    
        return reward, next_state, done
        
    def learn(self):
        '''Updates the networks after every ith step as defined by the update_frequency parameter'''
        for _ in range(self.learn_steps):
            # sample from replay
            (s, a, r, ns, d), idxs, is_weights  = self.memory.sample(self.replay_size)

            # critic loss optimizer
            na = self.actor_target(ns)
            q_target_next = self.critic_target(ns, na).detach()
            
            #normalize rewards
            r = (r - r.mean()) / (r.std() + np.finfo(np.float32).eps.item())

            q_target = r + (self.gamma * (1-d) * q_target_next) 
            q = self.critic(s, a)
            critic_loss = (torch.FloatTensor(is_weights) * F.mse_loss(q, q_target)).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()
            
            errors = torch.abs(q - q_target).data.numpy()
            # update priority
            for i in range(self.replay_size):
                idx = idxs[i]
            self.memory.update(idx, errors[i])

            # actor loss optimizer
            a = self.actor(s)
            q = self.critic(s, a)
            actor_loss = -q.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update
            self.critic_target = self.soft_update(self.critic_target , self.critic, self.tau)
            self.actor_target = self.soft_update(self.actor_target , self.actor, self.tau)
        return
    
    def plotter(self, scores, deque_length = 100, plot_graph = True ,save_plot = False , target_score = 0.5, solved = False):
        '''Plots the score
        scores           ([float]) : List of all the final scores
        deque_length (Int/[float]) : Length of score deque or score score deque to calculate deque length
        plot_graph          (Bool) : Displays score graph
        save_plot           (Bool) : Saves the plot to a .png file
        target_score         (Int) : Target score for the Network
        solved              (Bool) : Defines if evvironment is solved. If True, final score printed.
        '''    
        if isinstance(deque_length,int) == False:
            deque_length = len(deque_length)
            
        self.average_window = []
        for w1 in range(1,min(len(scores),deque_length)+1):
            self.average_window.append(np.mean(scores[:w1]))
        if len(scores)>100:
            for w2 in range(deque_length+1,len(scores)+1):
                self.average_window.append(np.mean(scores[w2-deque_length:w2]))
                
        clear_output()    
        for i in range(0,len(self.average_window),self.print_every):
            if i!=0:
                print('Episode {}\tAverage Scores: {:.2f}'.format(i, self.average_window[i]))
            
        if solved == True:
            c = 0
            for i in self.average_window:
                if i > target_score:
                    print('\nEnvironment solved (solved > 0.5) after at {} episodes!\tAverage Score: {:.2f}'.format(c-100,np.mean(scores[-100:])))
                    break
                else:
                    c+=1
            c = 0
            for i in self.average_window:
                if i > 1.0:
                    print('\nEnvironment solved (solved > 1.0) after at {} episodes!\tAverage Score: {:.2f}'.format(c-100,np.mean(scores[-100:])))
                    break
                else:
                    c+=1
        # Score Plot
        plt.figure(figsize=(8,5))
        plt.plot([*range(1,len(scores)+1)],scores,label='Score')
        plt.plot([*range(1,len(scores)+1)],self.average_window,color='red',label=f'Average Score')
        plt.plot([*range(1,len(scores)+1)],[target_score for i in range(len(scores))],color='black',label='Target Score')
        plt.title(f'Training Graph for {self.num_agents} agents')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.grid()
        plt.legend()

        # save plot
        if save_plot == True:
            plt.savefig('Images/TainedNetworkScores.png')
        # Display    
        plt.show()
        return 

    def save_checkpoint(self):
        '''Saves the trained model'''
        # Save Model
        torch.save(self.actor.state_dict(), 'TrainedModel/MADDPG_Actor.pth')
        torch.save(self.critic.state_dict(), 'TrainedModel/MADDPG_Critic.pth')
        # Save Model Network Structures
        ujson.dump({"hidden_layers":self.actor.hidden},open('TrainedModel/MADDPG_Actor_Network.json','w'))
        ujson.dump({"hidden_layers":self.critic.hidden},open('TrainedModel/MADDPG_Critic_Network.json','w'))
