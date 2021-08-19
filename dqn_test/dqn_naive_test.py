import gym
from torch.utils import tensorboard

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import numpy as np

from train import DQN_LHC

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.LHC_RL, max_episode_steps=None),
    agent_builder=None,
)

agent_specs = {
    "Agent-LHC": agent_spec,
    #"Agent-008": agent_spec,
}

env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/intersections/roundabout"],
    agent_specs=agent_specs,
)


env.action_space=gym.spaces.Discrete(4)
env.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,80,80), dtype=np.float32)

trainer = DQN_LHC(env,64,tensorboard_dir='/home/haochen/SMARTS_test_TPDM/tb_log/DQN_1')
trainer.train(100000)
trainer.save('/home/haochen/SMARTS_test_TPDM/dqn_model')