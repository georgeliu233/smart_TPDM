import gym

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from tqdm import trange
class SimpleAgent(Agent):
    def act(self, obs):
        return "keep_lane"

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
    agent_builder=SimpleAgent,
)

agent_specs = {
    "Agent-007": agent_spec,
    #"Agent-008": agent_spec,
}

env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/intersections/2lane_circle"],
    agent_specs=agent_specs,
)

agents = {
    agent_id: agent_spec.build_agent()
    for agent_id, agent_spec in agent_specs.items()
}
observations = env.reset()

for i in range(1000):
    agent_actions = {
        agent_id: agents[agent_id].act(agent_obs)
        for agent_id, agent_obs in observations.items()
    }
    observations, r, done, _ = env.step(agent_actions)
    if done['__all__']:
        print('reset!')
        observations = env.reset()
    # if i%100==0:
    #     print(observations['Agent-007'])