
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from envision.client import Client as Envision
from smarts.core.scenario import Scenario
from smarts.core.sumo_road_network import SumoRoadNetwork
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

def generate_trajs():
    scenarios = '/home/haochen/SMARTS/scenarios/left_turn'
    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=SumoTrafficSimulation(headless=True, auto_start=True),
        envision=Envision()
    )
    scenarios_iterator = Scenario.scenario_variations(
        [scenarios],
        list([]),
    )

    smarts.reset(next(scenarios_iterator))

    expert_obs = []
    expert_obs_next = []
    expert_terminals = []
    cars_obs = {}
    cars_obs_next = {}
    cars_terminals = {}

    prev_vehicles = set()
    done_vehicles = set()
    for i in range(100):
        smarts.step({})
        current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        _vehicle_states = [v.state for v in smarts.vehicle_index.vehicles]
        print(_vehicle_states[0])
        assert 1==0

def decode_map_xml(path):
    network = SumoRoadNetwork.from_file(path)
    graph = network.graph
    lanepoints = network.lanepoints
    nodes = graph.getNodes()
    print(lanepoints)
    #print(graph.getEdges())
    # for node in nodes:
    #     routes = node.getShape()
    #     infos = node.
    #     x,y = [rt[0] for rt in routes],[rt[1] for rt in routes]
    #     plt.scatter(x,y)
    # plt.savefig('/home/haochen/SMARTS_test_TPDM/test_routes_2lane.png')
def test_scenario(scenario_root):
    routes = Scenario.discover_routes(scenario_root)
    print(routes)
if __name__ == "__main__":
    #generate_trajs()
    decode_map_xml('/home/haochen/SMARTS/scenarios/intersections/2lane/map.net.xml')
    #test_scenario('/home/haochen/SMARTS/scenarios/left_turn')
