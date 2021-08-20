from pathlib import Path
import os

from smarts.sstudio import gen_scenario, gen_missions, gen_traffic
import smarts.sstudio.types as t
import numpy as np

scenario = os.path.dirname(os.path.realpath(__file__))

missions = [t.Mission(t.Route(begin=("edge-south-SN", 1, 30), end=("edge-west-EW", 0, 60)), start_time=2)]

impatient_car = t.TrafficActor(
    name="im_car",
    speed=t.Distribution(sigma=0.1, mean=0.5), #The speed distribution of this actor in m/s
    lane_changing_model=t.LaneChangingModel(impatience=1, cooperative=0.25),
    junction_model=t.JunctionModel(drive_after_red_time=1.5, drive_after_yellow_time=1.0, impatience=1.0),
)

patient_car = t.TrafficActor(
    name="car",
    speed=t.Distribution(sigma=0.1, mean=0.5),
    lane_changing_model=t.LaneChangingModel(impatience=0.2, cooperative=0.5),
    junction_model=t.JunctionModel(drive_after_yellow_time=1.0, impatience=0.5),
)

vertical_routes = [("north-NS", "south-NS"), ("south-SN", "north-SN")]
horizontal_routes = [("west-WE", "east-WE"), ("east-EW", "west-EW")]
turn_left_routes = [("south-SN", "west-EW")]
turn_right_routes = [("south-SN", "east-WE")]

routes = horizontal_routes + turn_right_routes
traffic = t.Traffic(flows=[t.Flow(route=t.Route(begin=(f"edge-{r[0]}", 0, "random"), end=(f"edge-{r[1]}", 0, "random")),
                    rate=10, actors={impatient_car: 0.5, patient_car: 0.5}) for r in routes for _ in range(2)]
                    + [t.Flow(route=t.Route(begin=(f"edge-{r[0]}", 1, "random"), end=(f"edge-{r[1]}", 1, "random")),
                    rate=10, actors={impatient_car: 0.5, patient_car: 0.5}) for r in horizontal_routes for _ in range(2)])

for seed in np.random.choice(100, 20, replace=False):
    gen_traffic(scenario, traffic, name=f"horizontal_{seed}", seed=seed)

# Agent Missions
gen_missions(scenario=scenario, missions=missions)

