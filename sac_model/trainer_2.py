import os
import time
import logging
import argparse
import json
import numpy as np
import tensorflow as tf
from gym.spaces import Box
from collections import deque
import math

from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer


if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


class Trainer:
    def __init__(
            self,
            policy,
            env,
            args,
            test_env=None,
            state_input=False,
            n_steps=0,
            lstm=False):
        if isinstance(args, dict):
            _args = args
            args = policy.__class__.get_argument(Trainer.get_argument())
            args = args.parse_args([])
            for k, v in _args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    raise ValueError(f"{k} is invalid parameter.")
        self.return_log,self.step_log = [],[]
        self.eval_log,self.eval_steps = [],[]
        self.state_input = state_input
        self._set_from_args(args)
        self._policy = policy
        self._env = env
        self.n_steps = n_steps
        self.lstm = lstm
        self._test_env = self._env if test_env is None else test_env
        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(
                shape=env.observation_space.shape)

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix))
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir)

        if args.evaluate:
            assert args.model_dir is not None
        self._set_check_point(args.model_dir)

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()


    def _set_check_point(self, model_dir):
        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=self._output_dir, max_to_keep=5)

        if model_dir is not None:
            assert os.path.isdir(model_dir)
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))

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
                dist_from_centers.append(-1)
                angle_errors.append(-1)
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

        # observations =  np.array(
        #     dist_from_centers + angle_errors+ego.position[:2].tolist()+[ego.speed,ego.steering]+distances,
        #     dtype=np.float32,
        # )
        observations =  np.array(
            dist_from_centers + angle_errors+ego.position[:2].tolist()+[ego.steering]+distances,
            dtype=np.float32,
        )
        assert observations.shape[-1]==15,observations.shape
        return observations

    def __call__(self):
        print('Start-training.....')
        
        if self._evaluate:
            self.evaluate_policy_continuously()

        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0

        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)


        obs = self._env.reset()
        if self.state_input:
            obs = self.observation_adapter(obs['Agent-LHC'])
        else:
            obs = obs['Agent-LHC'].top_down_rgb.data
        
        if self.lstm:
            buffer_queue = deque(maxlen=self.n_steps)
            for _ in range(self.n_steps):
                buffer_queue.append(obs)
            obs = np.array(list(buffer_queue))
        
        while total_steps < self._max_steps:
            if total_steps < self._policy.n_warmup:
                action = self._env.action_space.sample()
            else:
                action = self._policy.get_action(obs)
            
            choice_action = []
            MAX_SPEED = 10
            choice_action.append((action[0]+1)/2*MAX_SPEED)
            if action[1]<= -1/3:
                choice_action.append(-1)
            elif -1/3< action[1] <1/3:
                choice_action.append(0)
            else:
                choice_action.append(1)
            #print(choice_action)
            next_obs, reward, done, _ = self._env.step({
            "Agent-LHC":choice_action
            })

            # next_obs, reward, done, _ = self._env.step(action)
            done_events = next_obs["Agent-LHC"].events
            r = 0.0
            if done_events.reached_goal or (done["Agent-LHC"] and not done_events.reached_max_episode_steps):
                r = 1.0
            if done_events.collisions !=[]:
                r = -1.0
            #self.memory.append(state, action, r, next_state, done["Agent-LHC"])
            episode_return += r

            if self.state_input:
                next_obs = self.observation_adapter(next_obs['Agent-LHC'])
            else:
                next_obs = next_obs['Agent-LHC'].top_down_rgb.data

            if self._show_progress:
                self._env.render()
            episode_steps += 1
            # episode_return += reward
            total_steps += 1
            tf.summary.experimental.set_step(total_steps)

            done_flag = done['Agent-LHC']
            if (hasattr(self._env, "_max_episode_steps") and
                episode_steps == self._env._max_episode_steps):
                done_flag = False
            
            # print(obs)
            # print(action)
            # print(next_obs)
            # print(done_flag)
            if self.lstm:
                buffer_queue.append(next_obs)
                next_obs = np.array(list(buffer_queue))

            replay_buffer.add(obs=obs, act=action,
                              next_obs=next_obs, rew=r, done=done_flag)
            obs = next_obs

            if done['Agent-LHC'] or episode_steps == self._episode_max_steps:
                replay_buffer.on_episode_end()
                obs = self._env.reset()
                if self.state_input:
                    obs = self.observation_adapter(obs['Agent-LHC'])
                else:
                    obs = obs['Agent-LHC'].top_down_rgb.data 
                if self.lstm:
                    buffer_queue = deque(maxlen=self.n_steps)
                    for _ in range(self.n_steps):
                        buffer_queue.append(obs)
                    obs = np.array(list(buffer_queue))
                self.return_log.append(episode_return)
                self.step_log.append(total_steps)
                with open('/home/haochen/SMARTS_test_TPDM/log_tf2rl_lstm_2.json','w',encoding='utf-8') as writer:
                    writer.write(json.dumps([self.return_log,self.step_log],ensure_ascii=False,indent=4))

                n_episode += 1
                fps = episode_steps / (time.perf_counter() - episode_start_time)
                self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                    n_episode, total_steps, episode_steps, episode_return, fps))
                tf.summary.scalar(name="Common/training_return", data=episode_return)
                tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

            if total_steps < self._policy.n_warmup:
                continue

            if total_steps % self._policy.update_interval == 0:
                samples = replay_buffer.sample(self._policy.batch_size)
                with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                    self._policy.train(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32),
                        None if not self._use_prioritized_rb else samples["weights"])
                if self._use_prioritized_rb:
                    td_error = self._policy.compute_td_error(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32))
                    replay_buffer.update_priorities(
                        samples["indexes"], np.abs(td_error) + 1e-6)

            if total_steps % self._test_interval == 0:
                avg_test_return, avg_test_steps = self.evaluate_policy(total_steps)
                self.eval_log.append(avg_test_return)
                with open('/home/haochen/SMARTS_test_TPDM/log_tf2rl_lstm_test_2.json','w',encoding='utf-8') as writer:
                    writer.write(json.dumps(self.eval_log,ensure_ascii=False,indent=4))
                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes))
                tf.summary.scalar(
                    name="Common/average_test_return", data=avg_test_return)
                tf.summary.scalar(
                    name="Common/average_test_episode_length", data=avg_test_steps)
                tf.summary.scalar(name="Common/fps", data=fps)

            if total_steps % self._save_model_interval == 0:
                self.checkpoint_manager.save()

        tf.summary.flush()

    def evaluate_policy_continuously(self):
        """
        Periodically search the latest checkpoint, and keep evaluating with the latest model until user kills process.
        """
        if self._model_dir is None:
            self.logger.error("Please specify model directory by passing command line argument `--model-dir`")
            exit(-1)

        self.evaluate_policy(total_steps=0)
        while True:
            latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)
            if self._latest_path_ckpt != latest_path_ckpt:
                self._latest_path_ckpt = latest_path_ckpt
                self._checkpoint.restore(self._latest_path_ckpt)
                self.logger.info("Restored {}".format(self._latest_path_ckpt))
            self.evaluate_policy(total_steps=0)

    def evaluate_policy(self, total_steps):
        tf.summary.experimental.set_step(total_steps)
        if self._normalize_obs:
            self._test_env.normalizer.set_params(
                *self._env.normalizer.get_params())
        avg_test_return = 0.
        avg_test_steps = 0
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)
        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []
            obs = self._test_env.reset()
            
            if self.state_input:
                obs = self.observation_adapter(obs['Agent-LHC'])
            else:
                obs = obs['Agent-LHC'].top_down_rgb.data
            
            if self.lstm:
                buffer_queue = deque(maxlen=self.n_steps)
                for _ in range(self.n_steps):
                    buffer_queue.append(obs)
                obs = np.array(list(buffer_queue))

            avg_test_steps += 1
            for _ in range(self._episode_max_steps):
                action = self._policy.get_action(obs, test=True)
                choice_action = []
                MAX_SPEED = 10
                choice_action.append((action[0]+1)/2*MAX_SPEED)
                if action[1]<= -1/3:
                    choice_action.append(-1)
                elif -1/3< action[1] <1/3:
                    choice_action.append(0)
                else:
                    choice_action.append(1)
                #print(choice_action)
                next_obs, reward, done, _ = self._env.step({
                "Agent-LHC":choice_action
                })

                # next_obs, reward, done, _ = self._env.step(action)
                done_events = next_obs["Agent-LHC"].events
                r = 0
                if done_events.reached_goal or (done["Agent-LHC"] and not done_events.reached_max_episode_steps):
                    r = 1
                if done_events.collisions !=[]:
                    r = -1
                #self.memory.append(state, action, r, next_state, done["Agent-LHC"])
                episode_return += r
                if self.state_input:
                    next_obs = self.observation_adapter(next_obs['Agent-LHC'])
                else:
                    next_obs = next_obs['Agent-LHC'].top_down_rgb.data
                # next_obs, reward, done, _ = self._test_env.step(action)

                if self.lstm:
                    buffer_queue.append(next_obs)
                    next_obs = np.array(list(buffer_queue))
                avg_test_steps += 1
                if self._save_test_path:
                    replay_buffer.add(obs=obs, act=action,
                                      next_obs=next_obs, rew=r, done=done['Agent-LHC'])

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))
                elif self._show_test_progress:
                    self._test_env.render()
                episode_return += r
                obs = next_obs
                if done['Agent-LHC']:
                    obs = self._test_env.reset()
                    obs = obs['Agent-LHC'].top_down_rgb.data
                    break
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
                total_steps, i, episode_return)
            if self._save_test_path:
                save_path(replay_buffer._encode_sample(np.arange(self._episode_max_steps)),
                          os.path.join(self._output_dir, prefix + ".pkl"))
                replay_buffer.clear()
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            avg_test_return += episode_return
        if self._show_test_images:
            images = tf.cast(
                tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
                tf.uint8)
            tf.summary.image('train/input_img', images,)
        return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes

    def _set_from_args(self, args):
        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = (args.episode_max_steps
                                   if args.episode_max_steps is not None
                                   else args.max_steps)
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._evaluate = args.evaluate
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env.')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode')
        parser.add_argument('--n-experiments', type=int, default=1,
                            help='Number of experiments')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process')
        parser.add_argument('--save-model-interval', type=int, default=int(1e4),
                            help='Interval to save model')
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help='Interval to save summary')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results')
        parser.add_argument('--normalize-obs', action='store_true',
                            help='Normalize observation')
        parser.add_argument('--logdir', type=str, default='results',
                            help='Output directory')
        # test settings
        parser.add_argument('--evaluate', action='store_true',
                            help='Evaluate trained model')
        parser.add_argument('--test-interval', type=int, default=int(1e4),
                            help='Interval to evaluate trained model')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process')
        parser.add_argument('--test-episodes', type=int, default=5,
                            help='Number of episodes to evaluate at once')
        parser.add_argument('--save-test-path', action='store_true',
                            help='Save trajectories of evaluation')
        parser.add_argument('--show-test-images', action='store_true',
                            help='Show input images to neural networks when an episode finishes')
        parser.add_argument('--save-test-movie', action='store_true',
                            help='Save rendering results')
        # replay buffer
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level')
        return parser
