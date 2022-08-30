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

class OptimalZonation(Problem):
    def __init__(self,_G,sn_id,_branch_lodf, _prob_id,_consider_lodf=False, partition_size_upper = 40, partition_size_lower=0, min_nodes_partition = 1):
        self.G = _G
        self.subname_id = sn_id
        self.np_size_upper = partition_size_upper
        self.np_size_lower = partition_size_lower
        self.min_node_per_partition = min_nodes_partition
        self.branch_lodf = _branch_lodf
        self.consider_lodf = _consider_lodf

        # save the original graph of each utility for later evaluation purpose
        nx.write_gpickle(self.G, str(_prob_id)+'_Dec22.gpickle')

        super().__init__(n_var=len(list(self.G.edges)), n_obj=4, xl=0, xu=1, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        # First objective is to minimize the number of ACLs
        f1 = []
        f12 =[]
        threads_f1 = {}
        res_f1 = {}
        res_f12 = {}
        for d in range(x.shape[0]):
            threads_f1[d] = threading.Thread(target=self.compute_metric_cyber_infrastructure, args=(x[d], res_f1,res_f12, d))
            threads_f1[d].start()
        for d in range(x.shape[0]):
            threads_f1[d].join()
        for i in sorted(res_f1.keys()):
            f1.append(res_f1[i])
            f12.append(res_f12[i])

        # Second objective is to maximize the resilience
        f2 = []
        threads_f2 = {}
        res_f2 = {}

        f3 = []
        res_f3 = {}

        for d in range(x.shape[0]):
            threads_f2[d] = threading.Thread(target=self.compute_si, args=(x[d],res_f2,res_f3, d))
            threads_f2[d].start()
        for d in range(x.shape[0]):
            threads_f2[d].join()
        for i in sorted(res_f2.keys()):
            f2.append(1000/res_f2[i]) # since we want to maximize the security index we keep the metric in denominator and a constant in numerator
            f3.append(1/res_f3[i]) # for the LODF we want to make sure more critical lines are not clubbed into a single zone, hence metric in denominator

        #out["F"] = anp.column_stack([np.array(f1), np.array(f2), np.array(f3)])
        out["F"] = anp.column_stack([np.array(f1), np.array(f12),np.array(f2), np.array(f3)])

        # add the constraint
        g = []
        # the number of sub-graph based on the solution should be within upper and lower limit
        g_temp = np.zeros(x.shape[0])
        threads_g1 = {}
        res_g1 = {}
        res_g2 = {}
        res_g3 = {}
        res_g4 = {}
        for m in range(x.shape[0]):
            threads_g1[m] = threading.Thread(target=self.compute_constraint, args=(x[m], res_g1, res_g2, res_g3,res_g4, m))
            threads_g1[m].start()
        for m in range(x.shape[0]):
            threads_g1[m].join()
        for i in sorted(res_g1.keys()):
            g_temp[i] = res_g1[i]
        g.append(g_temp)

        g_temp = np.zeros(x.shape[0])
        for i in sorted(res_g1.keys()):
            g_temp[i] = res_g2[i]
        g.append(g_temp)

        # g_temp = np.zeros(x.shape[0])
        # for i in sorted(res_g1.keys()):
        #     g_temp[i] = res_g3[i]
        # g.append(g_temp)

        g_temp = np.zeros(x.shape[0])
        for i in sorted(res_g1.keys()):
            g_temp[i] = res_g4[i]
        g.append(g_temp)

        out["G"] = anp.column_stack(np.array(g))
        print(out)

    # ********CONSTRAINT DESCRIPTION*******************************
    # the three res1, res2, res3 and res4 returns the four constraints
    # Constraint 1 and 2: sets the upper and lower limit on the no. of security zones within a UCC and their substation
    # Constraint 3: sets the lower bound on the minimum number of substations within a security zone.
    # Constraint 4: ensures that there exist atleast one substation in a zone that has the UCC as its neighbour
    def compute_constraint(self,sol,res1,res2,res3,res4,ix):
        edge_index = np.where(sol == 1)
        temp_graph = self.G.copy()
        # get the edge index from original graph
        edge_list = list(self.G.edges)
        for i, edge in enumerate(edge_list):
            if i in list(edge_index[0]):
                temp_graph.remove_edge(edge[0], edge[1])
        subgraph = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]

        # constraint 1 in the paper with upper and lower bound
        res1[ix] = len(subgraph) - self.np_size_upper
        res2[ix] = -len(subgraph) + self.np_size_lower

        # constraint 2 in the paper with minimum number nodes in a security zone
        const = -1
        for sg in enumerate(subgraph):
            #if len(list(sg[1].nodes)) < self.min_node_per_partition:
            if len(list(sg[1].nodes)) < int(len(list(self.G.nodes))/10)+self.min_node_per_partition:
                const = 1
                break
        res3[ix] = const

        # constraint 3 ensures that there used to exist atleast one node in a zone connected to UCC in original graph
        const2 = -1
        for sg in enumerate(subgraph):
            count = 0
            for node in sg[1].nodes:
                if 'ucc' not in node:
                    nbrs = [n for n in self.G.neighbors(node)]
                    found = ['ucc' in sub for sub in nbrs]
                    if True in found:
                        count +=1
            if count == 0:
                const2 = 1000
                break
        res4[ix] = const2

    # ACL count as the metric
    def compute_metric_cyber_infrastructure(self,sol,res,res2,ix):
        edge_index = np.where(sol == 1)
        temp_graph = self.G.copy()
        # get the edge index from original graph
        edge_list = list(self.G.edges)
        for i, edge in enumerate(edge_list):
            if i in list(edge_index[0]):
                temp_graph.remove_edge(edge[0], edge[1])
        subgraph = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]
        firewalls = 0
        # the subgraph that includes the UCC are not zoned hence the number of firewalls will be as original no. of nodes
        # the subgraph that doesnt include the UCC will have one firewall for each zone
        for sg in enumerate(subgraph):
            found = ['ucc' in sub for sub in sg[1].nodes]
            if True in found:
                firewalls += (len(list(sg[1].nodes)) - 1)
            else:
                firewalls += 1

        # the number of ACL depends on various factors
        # ACLs associated with UCC will be reduced with zonation
        # as well as the ACLs associated with the substations
        ucc_acl_count = 0
        sub_acl_count = 0
        for sg in enumerate(subgraph):
            found = ['ucc' in sub for sub in sg[1].nodes]
            if True in found:
                # substraction of 1 is to remove 2it-self, multiplication of 2 is for both sub DMZ and OT,
                # first addition of 5 is for fixed policy in the firewall connected to the substation
                ucc_acl_count += (2*(len(list(sg[1].nodes)) - 1) + 5)
                # 6 is associated with 3 deny rule, and 3 permit rule.
                # permits are 1. access to DNP outstation 2. access to local DB 3. access to local Web Server
                sub_acl_count += 6*(len(list(sg[1].nodes)) - 1)
            else:
                # if the sub-graph doesnt have the UCC connected
                sub_acl_count += 6
                ucc_acl_count += 2

        res[ix] = firewalls
        res2[ix] = ucc_acl_count + sub_acl_count

    def compute_si(self,sol,res,res2,ix):
        edge_index = np.where(sol == 1)
        temp_graph = self.G.copy()
        # get the edge index from original graph
        edge_list = list(self.G.edges)
        for i, edge in enumerate(edge_list):
            if i in list(edge_index[0]):
                temp_graph.remove_edge(edge[0], edge[1])
        subgraph = [temp_graph.subgraph(c) for c in nx.connected_components(temp_graph)]

        si_total = 0
        lodf_total = 0
        for i in range(len(subgraph)):
            si_cluster = 0
            lodf_cluster = 0
            relay_info = {}
            if self.consider_lodf:
                relay_info['lodf'] = 0.0
            if 'Line' not in relay_info.keys():
                relay_info['Line'] = 0
            if 'Disconnect' not in relay_info.keys():
                relay_info['Disconnect'] = 0
            if 'Breaker' not in relay_info.keys():
                relay_info['Breaker'] = 0
            if 'Transformer' not in relay_info.keys():
                relay_info['Transformer'] = 0
            for node in subgraph[i].nodes():
                if node in self.subname_id.keys():
                    sub_relay_info = self.get_relays(self.subname_id[node])
                    relay_info['Line'] += sub_relay_info['Line']
                    relay_info['Disconnect'] += sub_relay_info['Disconnect']
                    relay_info['Breaker'] += sub_relay_info['Breaker']
                    relay_info['Transformer'] += sub_relay_info['Transformer']
                    if self.consider_lodf:
                        relay_info['lodf'] += sub_relay_info['lodf']
            if relay_info['Breaker'] != 0:
                si_cluster += (1 / (1 * relay_info['Breaker']))
            if relay_info['Disconnect'] != 0:
                si_cluster += (1 / (3 * relay_info['Disconnect']))
            if relay_info['Transformer'] != 0:
                si_cluster += (1 / (4 * relay_info['Transformer']))
            if relay_info['Line'] != 0:
                si_cluster += (1 / (2 * relay_info['Line']))

            si_total+=si_cluster
        if self.consider_lodf:
            if relay_info['lodf']!=0.0:
                lodf_cluster+=(1/(10 * relay_info['lodf']))
            lodf_total += lodf_cluster
        #compute the LODF metric associated with the line
        res[ix] = si_total
        res2[ix] = lodf_total

    def get_relays(self,substation_name):
        full_list = os.listdir('TX2000/Substations/')
        file = [nm for nm in full_list if substation_name in nm][0]
        # parse the file based on the substation name
        f = open('TX2000/Substations/' + file)
        data = json.load(f)
        vals = data['nodes']['$values']

        relay_info = {}
        if 'Line' not in relay_info.keys():
            relay_info['Line'] = 0
        if 'Disconnect' not in relay_info.keys():
            relay_info['Disconnect'] = 0
        if 'Breaker' not in relay_info.keys():
            relay_info['Breaker'] = 0
        if 'Transformer' not in relay_info.keys():
            relay_info['Transformer'] = 0
        if self.consider_lodf:
            relay_info['lodf'] = 0.0
        for val in vals:
            if 'Relay' in val['$type'] and 'RelayController' not in val['$type']:
                # print(val)
                controls = val['breakers']['$values']
                for control in controls:

                    if control['controlType'] == 'Line':
                        relay_info['Line'] += 1
                    if control['controlType'] == 'Disconnect':
                        relay_info['Disconnect'] += 1
                    if control['controlType'] == 'Breaker':
                        relay_info['Breaker'] += 1
                    if control['controlType'] == 'Transformer':
                        relay_info['Transformer'] += 1
                    if self.consider_lodf:
                        lodf_val = 0.0
                        try:
                            lf = self.branch_lodf
                            if '->' in control['busNumber']:
                                from_bus = control['busNumber'].split('->', 1)[0]
                                to_bus = control['busNumber'].split('->', 1)[1]
                                ckt_id = control['name'].split(' ', 1)[1].split('_')[2]
                                m = lf.loc[((lf['BusNum'] == int(from_bus)) & (lf['BusNum:1'] == int(to_bus)) & (lf['LineCircuit'] == str(ckt_id))),'LODF_metric']
                                #if m != pd.np.nan:
                                lodf_val = m.values[0]
                                if lodf_val == 0.0:
                                    lodf_val = lf.loc[((lf['BusNum'] == int(to_bus)) & (lf['BusNum:1'] == int(from_bus)) & (
                                                lf['LineCircuit'] == str(ckt_id))), 'LODF_metric'].values[0]
                                    if lodf_val is None:
                                        lodf_val = 0.0
                        except:
                            pass
                        #print(lodf_val)
                        relay_info['lodf'] += abs(lodf_val)
        return relay_info

withLodf = True
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
utilities = 150
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
            #if j >= 2 and j < 15:
            if random.random() > 0.7:
                G.add_edge(list(sg.keys())[0], k) # connect UCC
            else:
                if j >=2:
                    r = random.randint(1,j-1)
                    G.add_edge( k, list(sg.keys())[r]) # connect to a subs
                else:
                    G.add_edge(list(sg.keys())[0], k)
            # else:
            #     G.add_edge(list(sg.keys())[0], k)
        pos[k] = (v[0], v[1])

    # randomly add edges between graph nodes
    for m,node in enumerate(list(G.nodes)):
        if m > 2 and random.random() > 0.8:
            r = random.randint(1, m - 1)
            G.add_edge(node, list(G.nodes)[r])
    sg_info[i] = G

    # pos = nx.get_node_attributes(G,'pos')
    G_large = nx.compose(G_large, G)
    # nx.draw(G_large,pos,node_color = color_map,node_size=6)
# print(pos)
# print(G_large.nodes)
# nx.draw(G_large,pos,node_size=6)
# finally a huge graph is constructred

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



#nx.draw(G_large, pos, node_color=color_map, node_size=6)
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
                      zip(Branch['BusNum'].tolist(), Branch['BusNum:1'].tolist(), Branch['LineCircuit'].tolist())])
    # for computing the LODF matrix
    lodf_matrix = []
    lodf_params = ['LineLODF']
    i = 0
    for x, y, z in zip(Branch['BusNum'].tolist(), Branch['BusNum:1'].tolist(), Branch['LineCircuit'].tolist()):
        i += 1
        statement = "CalculateLODF(Branch %s %s %s)" % (x, y, z)
        SimAuto.RunScriptCommand(statement)
        lodf_result = SimAuto.GetParametersMultipleElement(ObjectType='branch', ParamList=lodf_params)['LineLODF'].tolist()
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

# formulate a problem
results={}
for k,sg in sg_info.items():
    nsga2_alg = NSGA2(
        pop_size=200,
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
        scipy.io.savemat('Utility_'+str(k)+'.mat', results)
