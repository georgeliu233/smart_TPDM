import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules import pooling
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F



USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


#PER-Implementation
class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def fusioned_push(self, state,img, action, reward, next_state,next_img, done):
        assert state.ndim == next_state.ndim
        assert img.ndim == next_img.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        img      = np.expand_dims(img, 0)
        next_img = np.expand_dims(next_img, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state,img, action, reward, next_state,next_img, done))
        else:
            self.buffer[self.pos] = (state,img, action, reward, next_state,next_img, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch       = list(zip(*samples))
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights
    
    def fusioned_sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch       = list(zip(*samples))
        states      = np.concatenate(batch[0])
        imgs      = np.concatenate(batch[1])
        actions     = batch[2]
        rewards     = batch[3]
        next_states = np.concatenate(batch[4])
        next_imgs = np.concatenate(batch[5])
        dones       = batch[6]

        return states,imgs, actions, rewards, next_states,next_imgs, dones, indices, weights
        
        
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)



class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions,env,dueling=False,fc_only=False,fusioned=False):
        super(CnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.env = env
        self.dueling=dueling
        self.fusioned=fusioned
        self.fc_only = fc_only
        
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 16, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.vehicle_features = nn.Sequential(
            nn.Linear(16,256),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512,256)
            #nn.Linear(512, self.num_actions)
        )
        self.fusion_norm = torch.nn.LayerNorm(256, eps=1e-05, elementwise_affine=True)

        self.fc_adv = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )

        self.fc_value = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        
    def forward(self, x):
        if not(self.fc_only or self.fusioned):
            x = x.view(-1,*self.input_shape)
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif self.fc_only:
            x = self.vehicle_features(x)
        elif self.fusioned:
            #x[0]:fc_input,x[1]:image_input
            feat = x[0]
            img = x[1]
            img = img.view(-1,*self.input_shape)
            img = self.features(img)
            img = img.view(img.size(0), -1)
            img = self.fc(img)
            feat = self.vehicle_features(feat)
            x = self.fusion_norm(img + feat)
            
        
        ac_scores = self.fc_adv(x)
        
        if self.dueling:
            value_scores = self.fc_value(x)
            #ac_scores_mean = torch.mean(ac_scores,dim=1)
            #ac_center = ac_scores - torch.unsqueeze(ac_scores_mean,1)
            output = value_scores + ac_scores - ac_scores.mean()
        else:
            output = ac_scores
        return output
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            if self.fusioned:
                with torch.no_grad():
                    state_in = Variable(torch.FloatTensor(np.float32(state[0])).unsqueeze(0))
                    img = Variable(torch.FloatTensor(np.float32(state[1])).unsqueeze(0))
                q_value = self.forward([state_in,img])
            else:
                with torch.no_grad():
                    state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
                q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.env.action_space.n)
        return action

