import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import os

import json
import pickle

#from torch.utils import tensorboard

from DQN_net import NaivePrioritizedBuffer,CnnDQN,Variable,USE_CUDA
from tensorboardX import SummaryWriter
print(USE_CUDA)
class DQN_LHC(object):
    def __init__(self,env,batch_size,gamma=0.99,replay_initial=1000,PER_size=50000,
        epsilon_start=1.0,epsilon_final=0.01,epsilon_decay=3000,beta_start=0.4,beta_steps=10000,
        update_target_interval=1000,tensorboard_interval=100,tensorboard_dir=None,dueling=False):
        super(DQN_LHC).__init__()
        
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_target_interval = update_target_interval
        self.dueling = dueling

        self.tensorboard_dir = tensorboard_dir
        self.tensorboard_interval = tensorboard_interval
        if self.tensorboard_dir is not None:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        #build Networks
        self.current_model = CnnDQN(self.env.observation_space.shape, self.env.action_space.n,self.env)
        self.target_model  = CnnDQN(self.env.observation_space.shape, self.env.action_space.n,self.env)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model  = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.0001)

        #PER and schedules:
        self.replay_initial = replay_initial 
        self.replay_buffer  = NaivePrioritizedBuffer(PER_size)
        self.beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_steps)

        self.epsilon_by_frame = lambda frame_idx: epsilon_final + \
            (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        

        #Init assgin curr-training network to target
        self.update_target(self.current_model,self.target_model)
    
        
    def update_target(self,current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def compute_td_loss(self,batch_size, beta):
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(batch_size, beta) 

        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action     = Variable(torch.LongTensor(action))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(done))
        weights    = Variable(torch.FloatTensor(weights))

        q_values      = self.current_model(state)
        next_q_values = self.target_model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        #detach(freeze gradient) the target_q_value
        loss  = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss  = loss.mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        return loss
    
    def train(self,training_steps):

        losses = []
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        state = state["Agent-LHC"].top_down_rgb.data/255
        action_choice = ["keep_lane","slow_down","change_lane_left","change_lane_right"]
        for frame_idx in range(1, training_steps + 1):
            
            epsilon = self.epsilon_by_frame(frame_idx)
            action = self.current_model.act(state, epsilon)
            agent_action = {
                "Agent-LHC":action_choice[action]
            }
            next_state, reward, done, _ = self.env.step(agent_action)
            next_state = next_state["Agent-LHC"].top_down_rgb.data/255
            reward = reward["Agent-LHC"]
            done = done["Agent-LHC"]

            self.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                state = self.env.reset()
                state = state["Agent-LHC"].top_down_rgb.data/255
                all_rewards.append(episode_reward)
                episode_reward = 0
                
            if len(self.replay_buffer) > self.replay_initial:
                beta = self.beta_by_frame(frame_idx)
                loss = self.compute_td_loss(self.batch_size, beta)
                losses.append(loss.item())
                if self.tensorboard_dir is not None:
                    self.writer.add_scalar('Loss',losses[-1],frame_idx)
                    self.writer.add_scalar('Episode_reward',all_rewards[-1],frame_idx)
                
            # if frame_idx % self.tensorboard_interval == 0 and self.tensorboard_dir is not None:
            #     self.writer.add_scalar()
            #     #plot(frame_idx, all_rewards, losses)
                
            if frame_idx % self.update_target_interval == 0:
                self.update_target(self.current_model, self.target_model)
            

    def eval(self,obs):
        with torch.no_grad():
            obs   = Variable(torch.FloatTensor(np.float32(obs)).unsqueeze(0))
        q_value = self.target_model.forward(obs)
        action  = q_value.max(1)[1].item()
        return action
        
    def save(self,save_path,save_PER=False):
        #saving the hyperparam-dict(optional:PER):
        hyper_dict={
            "batch_size":self.batch_size,
            "gamma":self.gamma,
            "update_target_interval":self.update_target_interval,
            "dueling":self.dueling
        }
        if save_PER:
            hyper_dict['PER']=self.replay_buffer
        print('Saving dict....')
        with open(save_path+'/hyper_dict.pkl','w') as writer:
            pickle.dump(hyper_dict,writer)
        
        #Saving model params
        print('Dict saved! Saving model-current...')
        torch.save(self.current_model,save_path+'/current_model.pkl')
        print('Saving model-target...')
        torch.save(self.target_model,save_path+'/target_model.pkl')


    def load(self,load_path,eval=False):
        if not eval:
            print('loading dict...')
            with open(load_path+'/hyper_dict.pkl','r') as reader:
                hyper_dict = pickle.load(reader)

            print(hyper_dict.keys())
            self.batch_size = hyper_dict["batch_size"]
            self.gamma = hyper_dict["gamma"]
            self.update_target_interval = hyper_dict["update_target_interval"]
            self.dueling = hyper_dict["dueling"]
            if "PER" in hyper_dict:
                print('PER found,loading...')
                self.replay_buffer = hyper_dict["PER"]
            
            print('loading curr model...')
            self.current_model = torch.load(load_path+'/current_model.pkl')
        else:
            print('Only prediction')
        
        self.target_model = torch.load(load_path+'/target_model.pkl')
        print('ALL loaded')
        
