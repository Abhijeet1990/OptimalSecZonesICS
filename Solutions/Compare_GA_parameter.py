# This is the code for the network zonation for optimal firewall configuration for synthetic nw
### extract geographic location of the substation
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
from collections import OrderedDict
import random
import os
#import json
import time
import scipy.io
from esa import SAW
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
import autograd.numpy as anp
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.configuration import Configuration
Configuration.show_compile_hint = False
import threading
#import simplejson as json
import simplejson as json
from Problems.OptimalZone import OptimalZonation

def compute_lodf():
    ## compute LODF
    branch_lodf = pd.DataFrame()
    if withLodf:
        SimAuto = SAW(
            FileName=r"D:\Experiments\pymoo-master\Exp\TX2000\ACTIVSg2000_ControlCenterExperience.pwb",
            CreateIfNotFound=True)
        params = SimAuto.get_key_fields_for_object_type('branch')['internal_field_name'].tolist()
        Branch = SimAuto.GetParametersMultipleElement(ObjectType='branch', ParamList=params)
        G_mult = nx.MultiGraph()
        G_mult.add_edges_from([tuple([x, y, {'LineCircuit': z, 'Send': x, 'Receive': y}]) for x, y, z in
                               zip(Branch['BusNum'].tolist(), Branch['BusNum:1'].tolist(),
                                   Branch['LineCircuit'].tolist())])
        # for computing the LODF matrix
        lodf_matrix = []
        lodf_params = ['LineLODF']
        i = 0
        for x, y, z in zip(Branch['BusNum'].tolist(), Branch['BusNum:1'].tolist(), Branch['LineCircuit'].tolist()):
            i += 1
            statement = "CalculateLODF(Branch %s %s %s)" % (x, y, z)
            SimAuto.RunScriptCommand(statement)
            lodf_result = SimAuto.GetParametersMultipleElement(ObjectType='branch', ParamList=lodf_params)[
                'LineLODF'].tolist()
            lodf_matrix.append(lodf_result)

        # Collect the investigated branches which is the ones that qualify for the metric (mean(LODF/std(LODF)))
        # Sort

        n_contingency = 3
        metric = []
        np_lodf = np.array(lodf_matrix)
        for row in np_lodf:
            metric_ = np.mean(row) / (np.std(row))
            metric.append(metric_)

        branch_lodf = pd.concat([Branch, pd.DataFrame(metric, columns=['LODF_metric'])],
                                axis=1)  # LODF_metric is already converted metric by mean and std
    return branch_lodf


def compute_complete_graph(nodes_whole_graph):
    G_large = nx.Graph()

    # store the sub-graphs for further zonation
    sg_info = {}

    pos = {}
    for i, sg in enumerate(nodes_whole_graph):
        G = nx.Graph()
        for j, (k, v) in enumerate(sg.items()):
            G.add_node(k, pos=(v[0], v[1]))

            # directly connect to the
            if 'ucc' not in k:
                # if j >= 2 and j < 15:
                if random.random() > 0.7:
                    G.add_edge(list(sg.keys())[0], k)  # connect UCC
                else:
                    if j >= 2:
                        r = random.randint(1, j - 1)
                        G.add_edge(k, list(sg.keys())[r])  # connect to a subs
                    else:
                        G.add_edge(list(sg.keys())[0], k)
                # else:
                #     G.add_edge(list(sg.keys())[0], k)
            pos[k] = (v[0], v[1])

        # randomly add edges between graph nodes
        for m, node in enumerate(list(G.nodes)):
            if m > 2 and random.random() > 0.8:
                r = random.randint(1, m - 1)
                G.add_edge(node, list(G.nodes)[r])
        sg_info[i] = G

        # pos = nx.get_node_attributes(G,'pos')
        G_large = nx.compose(G_large, G)
        # nx.draw(G_large,pos,node_color = color_map,node_size=6)
    return G_large, sg_info, pos


def compute_graph_with_lodf(_withLodf=True):
    withLodf = _withLodf
    data = pd.read_csv('TX2000/SubstationGeoAll.csv')
    # data
    # hypothetically assign utility control center location
    subname_id = {}

    location = []
    substationsList = []
    location_dict = {}
    for i, c in data.iterrows():
        location.append([c['Longitude'], c['Latitude']])
        substationsList.append(c['SubID'])
        if str(c['SubID']) in location_dict.keys():
            location_dict[str(c['SubID'])].append([c['Longitude'], c['Latitude']])
            # subname_id[str(c['SubID'])].append(str(c['SubName']))
        else:
            location_dict[str(c['SubID'])] = [[c['Longitude'], c['Latitude']]]
            subname_id[str(c['SubID'])] = str(c['SubName'])
    utilities = 100
    X = np.array(location)
    # perform the clustering
    kmeans = KMeans(n_clusters=utilities)
    kmeans = kmeans.fit(X)
    labels = kmeans.predict(X)
    print('label ' + str(len(labels)))
    print('location ' + str(len(location)))
    utilitiesGeoLoc = kmeans.cluster_centers_
    utilitiesLon = [lon[0] for lon in utilitiesGeoLoc]
    utilitiesLat = [lat[1] for lat in utilitiesGeoLoc]
    # labels
    # within each cluster construct graph keeping one UCC and the substations within the graph

    utilitiesSubstations = {}  # utility is the column of the list, rows are the substation numbers
    for sub in range(len(labels)):
        if labels[sub] in utilitiesSubstations.keys():
            utilitiesSubstations[labels[sub]].append(substationsList[sub])
        else:
            utilitiesSubstations[labels[sub]] = [substationsList[sub]]

    utilitiesSubstations = OrderedDict((k, v) for k, v in sorted(utilitiesSubstations.items(), key=lambda x: x))

    # connect the UCC of each cluster to the BA's
    nodes_whole_graph = []
    for i, (k, v) in enumerate(utilitiesSubstations.items()):
        nodes_in_cluster = {}
        nodes_in_cluster['ucc' + str(i + 1)] = utilitiesGeoLoc[i].tolist()
        for j in range(len(v)):
            nodes_in_cluster[str(v[j])] = location_dict[str(v[j])][0]
        nodes_whole_graph.append(nodes_in_cluster)

    G_large, sg_info, pos = compute_complete_graph(nodes_whole_graph)

    ba_nodes = ['prim_ba', 'sec_ba']
    G_large.add_node(ba_nodes[0], pos=(-101, 32))
    pos[ba_nodes[0]] = (-101, 32)
    G_large.add_node(ba_nodes[1], pos=(-102, 31.5))
    pos[ba_nodes[1]] = (-95, 31.5)
    color_map = []
    for i, n in enumerate(G_large.nodes()):
        if 'ucc' in str(n):
            G_large.add_edge(n, ba_nodes[0])
            G_large.add_edge(n, ba_nodes[1])
            G_large.nodes[n]['label'] = 'UCC'
            color_map.append('red')
        elif 'ba' in str(n):
            G_large.nodes[n]['label'] = 'BA'
            color_map.append('brown')
        else:
            G_large.nodes[n]['label'] = 'SUB'
            color_map.append('green')
    labels_G = nx.get_node_attributes(G_large, 'label')
    branch_lodf = compute_lodf()

    return subname_id, sg_info, branch_lodf


if __name__ == "__main__":
    withLodf = True
    subname_id, sg_info, branch_lodf = compute_graph_with_lodf(_withLodf=withLodf)
    # formulate a problem
    results={}

    # instead of analysing all the subgraph analyze the effect of cross over points on the solution for a given sub-graph
    for k,sg in sg_info.items():
        if len(sg.nodes()) > 10:
            for ps in range(50, 250, 50):
                nsga2_alg = NSGA2(
                    pop_size=ps,
                    n_offsprings=20,
                    sampling=get_sampling("bin_random"),
                    crossover=get_crossover("bin_k_point", prob=0.9, n_points=20),
                    mutation=get_mutation("bin_bitflip"),
                    eliminate_duplicates=True)
                gp_problem = OptimalZonation(_G=sg,sn_id=subname_id,_branch_lodf=branch_lodf, _prob_id=k,_consider_lodf=withLodf)
                start = time.time()
                res_gp_mo = minimize(gp_problem,
                                     nsga2_alg,
                                     termination=('n_gen', 100),
                                     seed=1,
                                     save_history=True)

                end = time.time()
                time_to_solve = end - start
                print('Parallel Solves in ' + str(time_to_solve) + ' secs')
                print('******Graph partition solution NSGA-2*************')
                print('Best soln found %s ' % res_gp_mo.X)
                print('Func Value %s ' % res_gp_mo.F)
                print('Constraint Value %s ' % res_gp_mo.G)
                if res_gp_mo.X is not None:
                    results['soln'] = res_gp_mo.X
                    results['score'] = res_gp_mo.F
                    results['time'] = time_to_solve
                    scipy.io.savemat('Utility_'+str(k)+'_size'+str(len(sg.nodes()))+'_ps'+str(ps)+'.mat', results)
            break
