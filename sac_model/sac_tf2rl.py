
import os
import gym
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='3'

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec,Agent

import sys
#sys.path.append('/home/haochen/SMARTS_test_TPDM/sac_model/sac_pic.py')
from tf2rl.experiments.trainer import Trainer
from sac_pic import SAC
#from tf2rl.algos.sac import SAC as SAC
agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.LanerWithSpeed, max_episode_steps=1000),
    agent_builder=None
)
agent_specs={
    'Agent-LHC':agent_spec
}
env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/left_turn"],
    agent_specs=agent_specs,
)

env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
env.observation_space = gym.spaces.Box(low=0, high=1, shape=(80,80,3), dtype=np.float32)


parser = Trainer.get_argument()
parser = SAC.get_argument(parser)
args = parser.parse_args()

args.max_steps=100000
#args.model_dir='/home/haochen/SMARTS_test_TPDM/sac_model/tf2rl_model'
#args.normalize_obs=True
args.logdir='/home/haochen/SMARTS_test_TPDM/sac_model/tf2rl_log'
args.test_episodes=10
args.use_prioritized_rb=True
args.use_nstep_rb=True
args.save_summary_interval=int(1e2)

policy = SAC(
    state_shape=(80,80,3),
    action_dim=2,
    auto_alpha=True,
    n_warmup=int(5e3),
    memory_capacity=int(2e4),
    batch_size=32
)

trainer = Trainer(policy,env,args,test_env=env)
trainer()