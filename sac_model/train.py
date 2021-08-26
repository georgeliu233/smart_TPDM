import os
import gym

os.environ['CUDA_VISIBLE_DEVICES']='2'

from sacd.agent import SacdAgent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.LHC_RL, max_episode_steps=1000),
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
env.action_space=gym.spaces.Discrete(4)
#env.observation_space = gym.spaces.Box(low=0, high=10, shape=(16), dtype=np.float32)

log_dir = '/home/haochen/SMARTS_test_TPDM/sac_model/sac_log'
agent = SacdAgent(env,test_env=None,log_dir=log_dir,num_steps=100000,batch_size=128,
                memory_size=20000,start_steps=1000,update_interval=4,target_update_interval=500,
                use_per=True,dueling_net=True,max_episode_steps=1000,multi_step=2)
agent.run()
#agent.save_models(os.path.join(agent.model_dir, 'final_model'))