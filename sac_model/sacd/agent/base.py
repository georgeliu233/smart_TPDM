from abc import ABC, abstractmethod
import os
import math
import numpy as np
import torch
from tensorboardX import SummaryWriter
import json
from sacd.memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from sacd.utils import update_params, RunningMeanStats


class BaseAgent(ABC):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, num_eval_steps=125000, max_episode_steps=27000,
                 log_interval=10, eval_interval=1000, cuda=True, seed=0,obs_dim=[16]):
        super().__init__()
        self.env = env
        self.test_env = test_env
        self.obs_dim  = obs_dim
        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        if self.test_env is not None:
            self.test_env.seed(2**31-1-seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        # LazyMemory efficiently stores FrameStacked states.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                capacity=memory_size,
                state_shape=self.obs_dim,
                device=self.device, gamma=gamma, multi_step=multi_step,
                beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                capacity=memory_size,
                state_shape=self.obs_dim,
                device=self.device, gamma=gamma, multi_step=multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)
        self.return_log = []
        self.step_log = []
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval


        self.action_choice = ["keep_lane","slow_down","change_lane_left","change_lane_right"]

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies, weights):
        pass
    
    def observation_adapter(self,env_obs):
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        # distance of vehicle from center of lane
        # closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

        dist_from_centers = []
        angle_errors = []
        if len(wps)<3:
            for _ in range(3-len(wps)):
                dist_from_centers.append(0.0)
                angle_errors.append(0.0)
        for wp in wps:
            signed_dist_from_center = wp.signed_lateral_error(ego.position)
            lane_hwidth = wp.lane_width * 0.5
            dist_from_centers.append(signed_dist_from_center / lane_hwidth)
            angle_errors.append(wp.relative_heading(ego.heading))
        


        neighborhood_vehicles = env_obs.neighborhood_vehicle_states
        relative_neighbor_distance = [np.array([10, 10])]*3

        # no neighborhood vechicle
        if neighborhood_vehicles == None or len(neighborhood_vehicles) == 0:
            relative_neighbor_distance = [
                distance.tolist() for distance in relative_neighbor_distance]
        else:
            position_differences = np.array([math.pow(ego.position[0]-neighborhood_vehicle.position[0], 2) +
                                            math.pow(ego.position[1]-neighborhood_vehicle.position[1], 2) for neighborhood_vehicle in neighborhood_vehicles])

            nearest_vehicle_indexes = np.argsort(position_differences)
            for i in range(min(3, nearest_vehicle_indexes.shape[0])):
                relative_neighbor_distance[i] = np.clip(
                    (ego.position[:2]-neighborhood_vehicles[nearest_vehicle_indexes[i]].position[:2]), -10, 10).tolist()

        distances = [
                diff for diffs in relative_neighbor_distance for diff in diffs]
        # print(len(dist_from_centers))
        # print(len(angle_errors))
        # print(len(ego.position[:2].tolist()))
        # print(len(distances))

        observations =  np.array(
            dist_from_centers + angle_errors+ego.position[:2].tolist()+[ego.speed, ego.steering]+distances,
            dtype=np.float32,
        )
        assert observations.shape[-1]==16,observations.shape
        return observations

    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = {"Agent-LHC":False}
        state = self.env.reset()
        state = self.observation_adapter(state["Agent-LHC"])

        while (not done["Agent-LHC"]) and episode_steps <= self.max_episode_steps:

            if self.start_steps > self.steps:
                action = self.env.action_space.sample()
            else:
                action = self.explore(state)
            

            next_obs, reward, done, _ = self.env.step({
                "Agent-LHC":self.action_choice[action]
            })
            next_state = self.observation_adapter(next_obs["Agent-LHC"])
            done_events = next_obs["Agent-LHC"].events
            if done_events.collisions !=[]:
                reward["Agent-LHC"] -= 10
            if done_events.wrong_way:
                reward["Agent-LHC"] -= 2
                
            # Clip reward to [-1.0, 1.0].
            #clipped_reward = max(min(reward, 1.0), -1.0)

            # To calculate efficiently, set priority=max_priority here.
            self.memory.append(state, action, reward["Agent-LHC"], next_state, done["Agent-LHC"])

            self.steps += 1
            episode_steps += 1
            episode_return += reward["Agent-LHC"]
            state = next_state

            if self.is_update():
                self.learn()

            if self.steps % self.target_update_interval == 0:
                self.update_target()

            if self.steps % self.eval_interval == 0 and self.test_env is not None:
                self.evaluate()
                self.save_models(os.path.join(self.model_dir, 'final'))

        # We log running mean of training rewards.
        self.train_return.append(episode_return)
        self.return_log.append(episode_return)
        self.step_log.append(self.steps)

        with open('/home/haochen/SMARTS_test_TPDM/log_sacd_left.json','w',encoding='utf-8') as writer:
            writer.write(json.dumps([self.return_log,self.step_log],ensure_ascii=False,indent=4))
        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_return.get(), self.steps)

        # print(f'Episode: {self.episodes:<4}  '
        #       f'Episode steps: {episode_steps:<4}  '
        #       f'Return: {episode_return:<5.1f}')

    def learn(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
            hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.use_per:
            self.memory.update_priority(errors)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                action = self.exploit(state)
                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __del__(self):
        self.env.close()
        if self.test_env is not None:
            self.test_env.close()
        self.writer.close()
