import os
import gym
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='3'

from sacd.agent import SacdAgent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.LanerWithSpeed, max_episode_steps=1000),
    agent_builder=None
)

agent_specs = {
    "Agent-LHC": agent_spec,
    #"Agent-008": agent_spec,
}

env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/left_turn"],
    agent_specs=agent_specs,
)

env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
#env.observation_space = gym.spaces.Box(low=0, high=255, shape=(80,80,3), dtype=np.int8)

log_dir = '/home/haochen/SMARTS_test_TPDM/sac_model/sac_log_con'
agent = SacdAgent(env,test_env=None,log_dir=log_dir,num_steps=100000,batch_size=32,
                memory_size=20000,start_steps=5000,update_interval=1,target_update_interval=1000,
                use_per=True,dueling_net=False,max_episode_steps=1000,multi_step=4,continuous=True,action_space=env.action_space.shape,
                obs_dim=(80,80,3),cnn=True,simple_reward=True,use_value_net=True,target_entropy_ratio=1)
agent.run()
agent.save_models('/home/haochen/SMARTS_test_TPDM/sac_model/sac_log_con/model2/')