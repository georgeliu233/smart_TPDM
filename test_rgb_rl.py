import gym

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from tqdm import trange
import matplotlib.pyplot as plt

import numpy as np
class SimpleAgent(Agent):
    def act(self, obs):
        act_set = ["change_lane_left","change_lane_right"]
        action = act_set[np.random.randint(0,2)]
        return action

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.LHC_RL, max_episode_steps=None),
    agent_builder=None,
)

agent_specs = {
    "Agent-007": agent_spec,
    #"Agent-008": agent_spec,
}

env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/intersections/roundabout"],
    agent_specs=agent_specs,
)
print('ac:')
print(env.action_space)
print('ob')
print(env.observation_space)
# agents = {
#     agent_id: agent_spec.build_agent()
#     for agent_id, agent_spec in agent_specs.items()
# }
observations = env.reset()

for i in range(100):
    # agent_actions = {
    #     agent_id: agents[agent_id].act(agent_obs)
    #     for agent_id, agent_obs in observations.items()
    # }
    #print(agent_actions["Agent-007"])
    act_set = ["change_lane_left","change_lane_right"]
    action = act_set[np.random.randint(0,2)]
    agent_actions={
        "Agent-007":action
    }
    observations, r, done, _ = env.step(agent_actions)
    print(r)
    obs = observations["Agent-007"]
    if i%20==0:
        print(obs.top_down_rgb.data.shape)
        plt.figure()
        plt.imshow(obs.top_down_rgb.data)
        plt.savefig('test_image/'+str(i)+'.png')
    if done['__all__']:
        print('reset!')
        observations = env.reset()
    # if i%100==0:
    #     print(observations['Agent-007'])