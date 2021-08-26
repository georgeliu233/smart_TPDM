import logging

import gym

from smarts.core.utils.episodes import episodes
#from examples import default_argument_parser


logging.basicConfig(level=logging.INFO)


def main(scenarios, headless, num_episodes, seed, max_episode_steps=None):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={},
        headless=headless,
        sumo_headless=True,
        visdom=False,
        seed=seed,
        timestep_sec=0.1,
    )

    if max_episode_steps is None:
        max_episode_steps = 1000

    for episode in episodes(n=num_episodes):
        env.reset()
        #episode.record_scenario(env.scenario_log)

        for _ in range(max_episode_steps):
            out = env.step({})
            print(out)
            #episode.record_step({}, {}, {}, {})

    env.close()


if __name__ == "__main__":
    #parser = default_argument_parser("egoless-example")
    #args = parser.parse_args()

    main(
        scenarios=['scenarios/left_turn'],
        headless=False,
        num_episodes=10,
        seed=66,
    )