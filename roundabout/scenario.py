import os
import random
from pathlib import Path
import numpy as np

from smarts.sstudio import gen_traffic, gen_missions, gen_social_agent_missions, gen_scenario
from smarts.sstudio.types import (
    Scenario,
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    SocialAgentActor,
    Distribution,
    LaneChangingModel,
    JunctionModel,
    Mission,
    EndlessMission,
)

scenario = os.path.dirname(os.path.realpath(__file__))

# Traffic Vehicles
#
cooperative_car = TrafficActor(
    name="cooperative_car",
    speed=Distribution(sigma=0.2, mean=0.6),
    lane_changing_model=LaneChangingModel(impatience=0.2, cooperative=0.5),
    junction_model=JunctionModel(drive_after_red_time=1.5, drive_after_yellow_time=1.0, impatience=0.5)
)

aggressive_car = TrafficActor(
    name="aggressive_car",
    speed=Distribution(sigma=0.2, mean=0.8),
    lane_changing_model=LaneChangingModel(impatience=1, cooperative=0.25),
    junction_model=JunctionModel(drive_after_yellow_time=1.0, impatience=1)
)

vertical_routes = [("north-NS", "south-NS"), ("south-SN", "north-SN")]

horizontal_routes = [("west-WE", "east-WE")]

turn_left_routes = [
    ("south-SN", "west-EW"),
    ("west-WE", "north-SN"),
    ("north-NS", "east-WE"),
    #("east-EW", "south-NS"),
]

turn_right_routes = [
    ("south-SN", "east-WE"),
    ("west-WE", "south-NS"),
    ("north-NS", "west-EW"),
    #("east-EW", "north-SN"),
]

turn_around_routes = [
    ("south-SN", "south-NS"),
    ("north-NS", "north-SN"),
    ("west-WE", "west-EW")
]

routes = vertical_routes + horizontal_routes + turn_left_routes + turn_right_routes + turn_around_routes

traffic = Traffic(flows=[Flow(route=Route(begin=(f"edge-{r[0]}", 0, "random"), end=(f"edge-{r[1]}", 0, "max")),
                  rate=10, actors={aggressive_car: 0.5, cooperative_car: 0.5}) for r in routes for _ in range(10)])

#print(len(traffic.flows))
for seed in np.random.choice(100,20,replace=False):
    gen_traffic(scenario, traffic, name=f"all_routes_{seed}", seed=seed)

# Agent Missions
gen_missions(scenario=scenario, missions=[Mission(Route(begin=("edge-east-EW", 0, 1), end=("edge-west-EW", 0, 10)), start_time=50)])
