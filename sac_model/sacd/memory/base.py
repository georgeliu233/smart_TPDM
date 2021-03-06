from collections import deque
import numpy as np
import torch


class MultiStepBuff:

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.reset()

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self, gamma=0.99):
        assert len(self.rewards) > 0
        state = self.states.popleft()
        action = self.actions.popleft()
        reward = self._nstep_return(gamma)
        return state, action, reward

    def _nstep_return(self, gamma):
        r = np.sum([r * (gamma ** i) for i, r in enumerate(self.rewards)])
        self.rewards.popleft()
        return r

    def reset(self):
        # Buffer to store n-step transitions.
        self.states = deque(maxlen=self.maxlen)
        self.actions = deque(maxlen=self.maxlen)
        self.rewards = deque(maxlen=self.maxlen)

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return len(self.rewards) == self.maxlen

    def __len__(self):
        return len(self.rewards)


class LazyMemory(dict):

    def __init__(self, capacity, state_shape, device,contiuous=False,action_shape=None,cnn=False):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self.contiuous=contiuous
        self.action_shape=action_shape
        self.cnn = cnn
        self.reset()

    def reset(self):
        self['state'] = []
        self['next_state'] = []
        if self.contiuous:
            self['action'] = []
        else:
            self['action'] = np.empty((self.capacity, 1), dtype=np.int64)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)

        self._n = 0
        self._p = 0

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        self['state'].append(state)
        self['next_state'].append(next_state)
        if self.contiuous:
            self['action'].append(action)
        else:
            self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        self.truncate()

    def truncate(self):
        while len(self['state']) > self.capacity:
            del self['state'][0]
            del self['next_state'][0]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0
        if self.cnn:
            ty = np.int8
        else:
            ty = np.float32
        states = np.empty(
            (batch_size, *self.state_shape), dtype=ty)
        next_states = np.empty(
            (batch_size, *self.state_shape), dtype=ty)
        
        #print(states.shape)

        for i, index in enumerate(indices):
            _index = np.mod(index+bias, self.capacity)
            states[i, ...] = self['state'][_index]
            next_states[i, ...] = self['next_state'][_index]

        if not self.cnn:
            states = torch.FloatTensor(states).to(self.device).float() #/ 255.
            next_states = torch.FloatTensor(
                next_states).to(self.device).float() #/ 255.
        else:
            states = np.ascontiguousarray(np.transpose(states,(0,3,1,2)),np.int32)
            states = torch.ByteTensor(states).to(self.device).float() #/ 255.
            next_states = np.ascontiguousarray(np.transpose(next_states,(0,3,1,2)),np.int32)
            next_states = torch.ByteTensor(
                next_states).to(self.device).float() #/ 255.

        if self.contiuous:
            actions = np.empty((batch_size, *self.action_shape), dtype=np.float32)
            for i, index in enumerate(indices):
                _index = np.mod(index+bias, self.capacity)
                actions[i, ...] = self['action'][_index]
            actions = torch.FloatTensor(actions).to(self.device)
        else:
            actions = torch.LongTensor(self['action'][indices]).to(self.device)

        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n


class LazyMultiStepMemory(LazyMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3,continuous=False,action_shape=None,cnn=False):
        super(LazyMultiStepMemory, self).__init__(
            capacity, state_shape, device,continuous,action_shape,cnn)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if self.buff.is_full():
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done:
                while not self.buff.is_empty():
                    state, action, reward = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done)
        else:
            self._append(state, action, reward, next_state, done)
