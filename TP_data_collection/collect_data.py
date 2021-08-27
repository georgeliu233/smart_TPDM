
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from envision.client import Client as Envision
from smarts.core.scenario import Scenario
from smarts.core.sumo_road_network import SumoRoadNetwork
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import trange

def generate_trajs(each_trajs=1000,steps=50000):

    scenarios = '/home/haochen/SMARTS/scenarios/left_turn'
    save_dir = "/home/haochen/SMARTS_test_TPDM/traj_data/"

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

    traj_num_cnt = 0
    # expert_obs = []
    # expert_obs_next = []
    # expert_terminals = []
    # cars_obs = {}
    # cars_obs_next = {}
    # cars_terminals = {}
    obs_dict = {}
    buf_obs_dict = {}

    exist_dict = {}
    prev_vehicles = set()
    curr_vehicles = set()

    traj_num = 0
    file_cnt = 0

    rst = 0
    for i in trange(steps):

        smarts.step({})

        #current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        _vehicle_states = [v.state for v in smarts.vehicle_index.vehicles]

        for vehicle_state in _vehicle_states:

            ve_id = vehicle_state.vehicle_id

            if ve_id not in buf_obs_dict:
                buf_obs_dict[ve_id]=[]
            curr_vehicles.add(ve_id)

            buf_obs_dict[ve_id].append((vehicle_state,i))
        
        if i>rst:
            #prev_step exist but current step does not exist = traj_done
            # print(prev_vehicles,curr_vehicles)
            # assert 1==0
            done_vehicles = prev_vehicles - curr_vehicles
            #print(buf_obs_dict)
            #print(done_vehicles)
            for v_id in done_vehicles:

                if v_id in obs_dict:
                    # in case we meet the same vehicle_id
                    if v_id not in exist_dict:
                        exist_dict[v_id] = 0
                    exist_dict[v_id] += 1
                    obs_dict[v_id+"_re_"+str(exist_dict[v_id])] = buf_obs_dict[v_id]
                else:    
                    obs_dict[v_id] = buf_obs_dict[v_id]
                
                del buf_obs_dict[v_id]
                
                traj_num += 1
                # if traj_num %20==0:
                #     print(traj_num)
                # save traj_files
                if traj_num >= each_trajs:
                    obs_dict.update(buf_obs_dict)
                    print('save file:',file_cnt)
                    with open(save_dir+"traj_1000_"+str(file_cnt)+'.pkl','wb') as writer:
                        pickle.dump(obs_dict,writer)

                    file_cnt += 1
                    traj_num = 0
                    obs_dict = {}
                    buf_obs_dict = {}
                    print('RESET')
                    smarts.reset(next(scenarios_iterator))
                    rst = i+1



        prev_vehicles = curr_vehicles
        curr_vehicles = set()    
        #print(_vehicle_states[0])

    with open(save_dir+"_"+str(file_cnt)+'.pkl','wb') as writer:
        pickle.dump(obs_dict,writer)
        

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
    generate_trajs()
    #decode_map_xml('/home/haochen/SMARTS/scenarios/intersections/2lane/map.net.xml')
    #test_scenario('/home/haochen/SMARTS/scenarios/left_turn')
